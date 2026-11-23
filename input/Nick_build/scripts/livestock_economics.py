#!/usr/bin/env python
"""Build LUTO's 2021-base livestock economics (agec_lvstk.h5) and the interim water delivery price (water_delivery_price.h5).

Sources (public, downloaded at run time, sha256-checked against pinned values):
  - ABS 6401.0 Consumer Price Index, Aug 2026 (All groups CPI, Australia, A2325846C): deflator to FY2021 dollars.
  - ABS 7215.0 Livestock products, June 2026: beef and veal, lamb and mutton carcase tonnes (Australia).
  - ABARES Agricultural commodities, Sep 2026 statistical tables 8 and 13: live cattle and sheep exports (number, $m FOB).
  - DAFF live animal export statistics (all livestock exports by state of export port), 2016-2021 and 2021-2026 files.
    Pinned on the rows and columns used, not the file bytes (DAFF files only, as for the LongPaddock grids in
    pasture_productivity.py): sha256 of the parsed table's column names (layout) and of the type and value of every cell
    used, i.e. cattle and sheep rows in the window (FY2017-21 from the 2016-2021 file, Jan 2021 onwards from the
    2021-2026 file, through FY2025), columns Month, Year (or Fin Year), State, Species and Quantity. Entirely blank rows
    and columns are dropped before the header row is found; a blank cell inside the used rows is hashed as NaN, so it
    counts as a change. DAFF appends a month to the 2021-2026 file each month (Sep 2026 rows appended by 30 Sep 2026,
    earlier rows identical). The raw-file sha256 is recorded but not enforced; any change in a used value, the layout
    or a cell type stops the run and sets the download aside.
  - ABS 7503.0 Value of agricultural commodities produced FY2017-22 and ABS Australian Agriculture: livestock FY2023-25
    (local value of cattle and sheep disposals, wool and milk, by state).
  - ABARES Farm Data Portal state and national historical files (farm costs and receipts, All Broadacre and Dairy).
  - ABS 4618.0 Water use on Australian farms FY2017-21 (volume applied to grazed pasture, by state).
  - AWI AWPFC wool production forecast reports and Dairy Australia "In Focus 2025": fetched and pinned only; the values
    used (wool and milk by state, wool cut per head) are hand-checked entries of the reference table below. AWI and
    Dairy Australia content is all rights reserved: fetched and pinned, not for redeposit.
  - FutureBeef "Water requirements for cattle" page: drinking water constants in the reference table, with the page URL
    and the sha256 of a saved copy; not fetched (terms unclear).
Inputs from the other 2021-base tools: the livestock map folder (livestock_map.py: lumap.h5, lmmap.h5,
lvstk_k.h5, livestock_recon_state_type.csv, livestock_recon_sa2.csv, livestock_dse_ratings.csv), the pasture layer
(pasture_productivity.py) and the SA2 2021 cell key (nlum_map.py). Inherited 2010-build inputs from
--input (inherited_2010/, copies of 2010-build files), each sha256 recorded in the provenance file: state_id.npy, real_area.h5, safe_pur_natl.h5,
safe_pur_modl.h5, NLUM_2010-11_mask.tif, agec_lvstk.h5 (column list and layout only), water_delivery_price.h5 (the
interim price values) and yieldincreases_bau2022.csv (the productivity input, file sha256 pinned). The outputs are therefore
not built from public sources alone.

Method:
  1. Window FY2017-25 (FY2017-21 for water), FY2021 dollars by the CPI (financial-year mean of four quarters).
  2. Prices (national): beef meat = (cattle disposals local value - live cattle export value) / 7215 beef and veal t;
     live cattle = export value / (number x weight); sheep likewise; wool = wool local value / AWPFC greasy t; milk = milk
     local value / Dairy Australia litres. Live-export weight per animal is carried from the 2010 files.
  3. State outputs: meat value = state disposals value - state live-export value (DAFF head share x national value per
     head), floored at 0, rescaled to the national meat value; wool and milk by the state shares of the reference table.
     Per NIR head (livestock map reconciliation); F = 1, Q = output per head; states without data take the national value.
     NT beef then has live-export revenue only.
  4. Costs: Farm Data Portal per head (beef and sheep split by DSE on All Broadacre farms, dairy on Dairy farms), per NIR
     head, x the share of livestock receipts the modelled products cover; national share where unitemised
     receipts exceed 10 per cent of a state's broadacre receipts, flagged "cause unverified" (NT). QC per head;
     AC, FLC, FOC, FDC per ha = rate x base-year head per ha on cells of the type, else potential dry head per ha.
  5. Water: WR_DRN from the drinking-water constants; WR_IRR = ABS grazed-pasture water by state over irrigated head by
     DSE, per-DSE factor within [1/3, 3] x national.
  6. water_delivery_price.h5: INTERIM: the 2010-build values, state median on blank cells, national median
     where a state has none. The water licence price is built by water_licence_price.py.
  7. yieldincreases_bau2022.csv: the 2010-build productivity input with a flat wool per-head trend, every wool column (dry and
     irrigated, modified and natural sheep land) 1 in every year, a change to default behaviour (the 2010-build file fell
     0.77 per cent a year dryland and 1.53 irrigated from 2010, source not recorded); AWPFC greasy cut per head is flat,
     -0.04 per cent a year FY2004-FY2024 (reference table). A text edit of the wool fields only; every other byte is kept.
Outputs in --out: agec_lvstk.h5, water_delivery_price.h5 (layout of the 2010-base files); yieldincreases_bau2022.csv (layout
of the 2010-base file); livestock_prices.csv, livestock_outputs_per_head.csv, livestock_costs_per_head.csv,
livestock_water_per_head.csv, receipts_share_report.csv (receipts share report), interim_items_livestock_economics.csv;
agec_lvstk.nc (AGEC_LVSTK over (column, cell), levels field and animal) and water_delivery_price.nc (WP over cell), NetCDF
copies in the jinzhu layout (nc_layout.py); provenance.csv (last).

Run as a plain script from the repository root (no luto imports):

    python input/Nick_build/scripts/livestock_economics.py --input input/Nick_build/inherited_2010 --livestock-map input/Nick_build/livestock_map
        --pasture input/Nick_build/pasture_productivity/pasture_kg_dm_ha.h5 --sa2 input/Nick_build/nlum_map/sa2_2021_cells.npy
        --out input/Nick_build/livestock_economics --cache input/Nick_build/_cache [--offline] [--source NAME=PATH]

Dependencies: numpy, pandas, PyTables, openpyxl, xlrd, rasterio, scipy, xarray, netCDF4. No luto imports (nc_layout.py
is a sibling module).
Memory: peak resident memory about 6.2 GB on the full grid (measured October 2026); run on a machine with at least 16
GB of RAM.
Output values are deterministic; HDF5 file bytes carry a group timestamp, so provenance records dataset content hashes.
"""
import argparse
import csv
import hashlib
import io
import os
import re
import sys
import time
import urllib.request
import zipfile
from importlib import metadata

import numpy as np
import pandas as pd
import rasterio

import nc_layout as NC

# ----------------------------------------------------------------------------- sources and pins
ABS7215 = 'https://www.abs.gov.au/statistics/industry/agriculture/livestock-products/jun-2026/{n}.xlsx'
ABS7503 = 'https://www.abs.gov.au/statistics/industry/agriculture/value-agricultural-commodities-produced-australia/'
ABSAAL = 'https://www.abs.gov.au/statistics/industry/agriculture/australian-agriculture-livestock/'
ABS4618 = 'https://www.abs.gov.au/statistics/industry/agriculture/water-use-australian-farms/'
AWI = 'https://www.wool.com/globalassets/wool/market-intelligence/wool-production-forecasts/'
DAFFD = 'https://www.agriculture.gov.au/sites/default/files/documents/'
SOURCES = {   # name: (url, sha256 of the raw file, note)
    'abs_cpi': ('https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/consumer-price-index-australia/'
                'aug-2026/CPI-Quarterly.zip', '0b97b11375f6cdbf87b59a85d9c4cd990950e43d09aeea4504221f72d9af3cbd',
                'ABS 6401.0 CPI Aug 2026, quarterly zip (6401018.xlsx), CC BY 4.0'),
    'abares_agcomm': ('https://daff.ent.sirsidynix.net.au/client/en_AU/search/asset/1038521/3',
                      '971ab4702970b1297981880374a6ef73993fb2552ea194dfe84c575b7e1cf4f3',
                      'ABARES Agricultural commodities Sep 2026 statistical tables, CC BY 4.0'),
    'abs_7215009': (ABS7215.format(n='7215009'), '8efa44249b9c59e5c9b8244f41bc7664f3376a5fedfa0ce30ba79315a7043941',
                    'ABS 7215.0 Jun 2026, meat produced, cattle (excl. calves), CC BY 4.0'),
    'abs_7215010': (ABS7215.format(n='7215010'), '014e5b530d0180b454c39c6e2d44de769b97f9d4c829250c32c7fbd81935aef9',
                    'ABS 7215.0 Jun 2026, meat produced, calves, CC BY 4.0'),
    'abs_7215011': (ABS7215.format(n='7215011'), '956a4a8764174167ba8b60fff68f36f7b5b741b30e96c7555a117ca9ea8e280e',
                    'ABS 7215.0 Jun 2026, meat produced, sheep, CC BY 4.0'),
    'abs_7215012': (ABS7215.format(n='7215012'), 'd0cb9d370a041271b921a2c48e11d938ee087d2f5c8acb67e8cbda083f48bd4b',
                    'ABS 7215.0 Jun 2026, meat produced, lambs, CC BY 4.0'),
    'daff_2016_2021': (DAFFD + 'all-livestock-exports-2016-2021.xlsx',
                       'ac397a49d993ccacaa2eae81129c4d3e5cc410ab7996795aede5d4fad3137583',
                       'DAFF all livestock exports 2016-2021 (data as at 31 Jul 2021), CC BY 4.0'),
    'daff_2021_2026': (DAFFD + 'all-livestock-exports-2021-2026.xlsx',
                       'f60049356adabd9323eef22c0c736056da6bd89049a5098d6bc2dd481116124a',
                       'DAFF all livestock exports 2021-2026 (data as at 31 Aug 2026; updated monthly), CC BY 4.0'),
    'abs_7503_2017': ('https://www.abs.gov.au/AUSSTATS/subscriber.nsf/log?openagent&75030do001_201617.xls&7503.0&'
                      'Data%20Cubes&1D4B9105890F1A41CA2582910013AA02&0&2016-17&21.05.2018&Previous',
                      '90fab9cc19e86bc52f285da320c239a3e6482c2b66df0bba6f17b84eb7062727', 'ABS 7503.0 2016-17 xls, CC BY 4.0'),
    'abs_7503_2018': (ABS7503 + '2017-18/75030do003_201718.csv',
                      '1f5c2cd7f55b25d85f5427c01bffb2dafdaa1579931ccdc1612482e614efe0d1', 'ABS 7503.0 2017-18 csv, CC BY 4.0'),
    'abs_7503_2019': (ABS7503 + '2018-19/75030do003_201819.csv',
                      'e9b99a53abdb5947cef7e541f176044dfa9699f1d0dd8d202ac1d6e48f0c8f42', 'ABS 7503.0 2018-19 csv, CC BY 4.0'),
    'abs_7503_2020': (ABS7503 + '2019-20/75030DO001_201920.xlsx',
                      '6adeb1b0931a8da0240e1eaed38626e097ad78f1007b1cf4def35e44d397aaec', 'ABS 7503.0 2019-20 xlsx, CC BY 4.0'),
    'abs_7503_2021': (ABS7503 + '2020-21/VACPDCASGS202021.xlsx',
                      'f737632f72d77f36cdff601487f2618cde703adb818727c6e5f7378209ea7eab', 'ABS 7503.0 2020-21 xlsx, CC BY 4.0'),
    'abs_7503_2022': (ABS7503 + '2021-22/VACPDCNAT_STATE202122.xlsx',
                      'cade129e4061b00361c1f612ad8a2ebcc43891ee1e7cfeb153987fdf413287d3', 'ABS 7503.0 2021-22 xlsx, CC BY 4.0'),
    'abs_aal_2023': (ABSAAL + '2022-23/AALDC_Value%20of%20livestock%20and%20products.xlsx',
                     'f37a0b3c9fee9472d5118681954802e041b020bbda8a365520896b834e62aa8b',
                     'ABS Australian Agriculture: Livestock 2022-23, value of livestock and products, CC BY 4.0'),
    'abs_aal_2024': (ABSAAL + '2023-24/AALDC_Value%20of%20livestock%20and%20products%202023-24.xlsx',
                     '9380a74189d5fa08017ffed5021ad97f511f5fb80cabfffa243da5d628f89241',
                     'ABS Australian Agriculture: Livestock 2023-24, value of livestock and products, CC BY 4.0'),
    'abs_aal_2025': (ABSAAL + '2024-25/AALDC_Value%20of%20livestock%20and%20products%202024-25.xlsx',
                     '6bd3cf8d7c3dc16527cf87c729948f3591b05ec66780b42d05a1bcd8ec1b11cd',
                     'ABS Australian Agriculture: Livestock 2024-25, value of livestock and products, CC BY 4.0'),
    'fdp_state': (DAFFD + 'fdp-state-historical.csv', '65d31d827532ad8246277596202a2eb326ab23d0b53f7c1815905c8f828f828b',
                  'ABARES Farm Data Portal state historical (release 30 Jun 2026), CC BY 4.0'),
    'fdp_national': (DAFFD + 'fdp-national-historical.csv', 'ffb28831b9b8e9ff658381cb0761f9b8ea3817204c02cf075b356955271edad0',
                     'ABARES Farm Data Portal national historical (release 30 Jun 2026), CC BY 4.0'),
    'abs_4618_2017': ('https://www.abs.gov.au/AUSSTATS/subscriber.nsf/log?openagent&46180do002_201617.xls&4618.0&'
                      'Data%20Cubes&3AB49B0730AFF6D2CA2582910013A4F1&0&2016-17&21.05.2018&Latest',
                      '3f5ac8615b081aad60cd44e772825c02d564471d97caf5e45ee22cddc931513a', 'ABS 4618.0 2016-17 SA4 xls, CC BY 4.0'),
    'abs_4618_2018': (ABS4618 + '2017-18/46180do004_201718.csv',
                      '0a4dd0aef3367587dd94f7c433ac7c9c564f7a8b218b73aaee4788930c3acbdb', 'ABS 4618.0 2017-18 ASGS csv, CC BY 4.0'),
    'abs_4618_2019': (ABS4618 + '2018-19/46180do004_201819.csv',
                      '1fd573f409cf1fd418c7385eb788c48397e5fa26699accd821cc131f889ac4db', 'ABS 4618.0 2018-19 ASGS csv, CC BY 4.0'),
    'abs_4618_2020': (ABS4618 + '2019-20/46180DO001_201920.xlsx',
                      '07dc36e1d3dbb254e89c8c7e3e5afed233e9007c26205567dadcb1248fa42e51', 'ABS 4618.0 2019-20 SA4 xlsx, CC BY 4.0'),
    'abs_4618_2021': (ABS4618 + '2020-21/WUAFDCASGS202021.xlsx',
                      '1453aacfcb132edb1609b36aad157530d02e2da3387702892b34f2368d6cd2eb', 'ABS 4618.0 2020-21 ASGS xlsx, CC BY 4.0'),
    'awi_aug2018': (AWI + 'australian-wool-production-forecast-report-august-2018.pdf',
                    '7706c9de8f1a2a4c0515f989a87e10427ca1e85e8cf275167833537ec1e055ad', 'AWI AWPFC Aug 2018, all rights reserved'),
    'awi_aug2019': (AWI + 'australian-wool-production-forecast-report-august-2019.pdf',
                    '17ad16df4166fb756a5924040901f27358ea944619b3bb14094d98a50801e835', 'AWI AWPFC Aug 2019, all rights reserved'),
    'awi_aug2020': (AWI + 'australian-wool-production-forecast-report-august-2020.pdf',
                    'd7c2aee7aac59be7268367db078d99de69eb761179216f8abd64457621d7e3db', 'AWI AWPFC Aug 2020, all rights reserved'),
    'awi_aug2021': (AWI + 'australian-wool-production-forecast-report---august-2021-amended.pdf',
                    '27c5e088ad086b2ccd278ce9d26756c3dc4b11193e4ca59350a600a882e5fbb0', 'AWI AWPFC Aug 2021 (amended), all rights reserved'),
    'awi_sep2022': (AWI + 'australian-wool-production-forecast-report---september-2022.pdf',
                    '640e489c71c820aac22d7a47c7f43f6da51e32a97a72e9519ba38eec4159a3b0', 'AWI AWPFC Sep 2022, all rights reserved'),
    'awi_aug2023': (AWI + 'australian-wool-production-forecast-report---august-2023.pdf',
                    '4861c618a461e28fbfec79190c84312d8eb743ceb0cabc28bc7999cbedb6ae5c', 'AWI AWPFC Aug 2023, all rights reserved'),
    'awi_aug2024': (AWI + '2024/august/australian-wool-production-forecast-report---august-2024.pdf',
                    '77412b93f981c355c734ec3770eb23270d9373e09afbe65d42b53fd97045278f', 'AWI AWPFC Aug 2024, all rights reserved'),
    'awi_aug2025': (AWI + '2025/august/australian-wool-production-forecast-report---august-2025.pdf',
                    '673f296b4bfefca2a905385a6efb87eedd7da240aa10be3610f9b82a8a4eb135', 'AWI AWPFC Aug 2025, all rights reserved'),
    'da_in_focus_2025': ('https://assets.dairyaustralia.com.au/api/public/content/In-Focus-2025',
                         'b7420ce1b9be4ce9cf1134bf3b922023bc260d0fc423958d7680f7cb012dc488',
                         'Dairy Australia, The Australian Dairy Industry In Focus 2025 (2,025,718 B), all rights reserved'),
}
# DAFF files: sha256 of the rows and columns used (enforced, see the docstring); the SOURCES sha256 is the raw file retrieved
# at pinning (recorded and tried first in the cache, not enforced)
CONTENT_PINS = {'daff_2016_2021': '9b92521bf080bd827169717af933ac63de3314d2429774d5a779ea7cce6d9a29',
                'daff_2021_2026': 'b3b6df950ecd2b363243600f63af6ad7f483a5333ef66522040e66fa7a291c6f'}
INHERITED =['state_id.npy', 'real_area.h5', 'safe_pur_natl.h5', 'safe_pur_modl.h5', 'NLUM_2010-11_mask.tif',
             'agec_lvstk.h5', 'water_delivery_price.h5', 'yieldincreases_bau2022.csv']
# the 2010-build productivity input whose wool columns are made flat (file sha256, pinned)
YIELDINC_SHA = '891759b2cd3927324cb97238368eedc251d39709ee158ce1471784695ae40fad'
WOOL_COLS = ('SHEEP - MODIFIED LAND WOOL', 'SHEEP - NATURAL LAND WOOL')
NCELL = 6956407

# ============================================================================= reference table (hand-entered)
# Each entry with its source, page and hash where it has one. Changing any value is a ruling.
REF = {
    # Window and price year; water window FY2017-21 (the ABS 4618 years available)
    'fy': list(range(2017, 2026)), 'fy_water': (2017, 2021), 'cpi_series': 'A2325846C', 'cpi_base_fy': 2021,
    # DSE per head (NIR-intake ratings of livestock_map.py, rounded to 6 dp): cost split, WR_IRR, pot head
    'dse': {'BEEF': 6.670396, 'SHEEP': 0.887382, 'DAIRY': 12.678383},
    # LUTO grassfed factors, unchanged from the 2010 build
    'grassfed': {'BEEF': .85, 'SHEEP': .85, 'DAIRY': .65},
    # live-export weight, t live weight per exported animal, carried from the 2010 files (Q3 of the 2010-base
    # agec_lvstk.h5), no public series. Label: "2010 basis, carried".
    'live_export_t_per_head': {'BEEF': 0.337144, 'SHEEP': 0.048389},
    # FutureBeef "Water requirements for cattle", https://futurebeef.com.au/resources/water-requirements/ (page modified
    # 2026-06-17; saved copy sha256 1badccc37eb3fac42f20d3c3108664e66cf97e96dfb297f623d573fbc0b1ad1e, 416,962 B; table
    # "Average daily consumption (L)", adapted from Burton 1965): beef cattle 45, dairy cows in milk 70, mature sheep on
    # dry pastures 7 L/head/day. Terms of use unclear.
    'drink_l_head_day': {'BEEF': 45.0, 'SHEEP': 7.0, 'DAIRY': 70.0},
    # Band on the per-DSE pasture water factor; unitemised-receipts threshold
    'water_band': (1 / 3, 3), 'unitemised_max': 0.10,
    # Farm Data Portal cost categories (variable names verbatim)
    'QC_COMMON': ['AI stud fees and herd testing ($)', 'Agistment ($)', 'Contracts - livestock ($)', 'Fodder ($)',
                  'Livestock materials ($)', 'Vet fees ($)'],
    'AC': ['Fertiliser ($)', 'Seed ($)', 'Crop and pasture chemicals ($)', 'Fuel oil and grease ($)', 'Repairs and maintenance ($)',
           'Electricity ($)', 'Other materials ($)', 'Other services ($)'],
    'FLC': ['Wages for hired labor ($)', 'Imputed labor cost ($)', 'Payments to sharefarmers ($)'],
    'FOC': ['Insurance ($)', 'Shire and PPB rates ($)', 'Accounting services ($)', 'Advisory services ($)', 'Telephone ($)',
            'Other administration expenses ($)', 'Leasing charges ($)', 'Stores and rations ($)'],
    'FDC': ['Depreciation ($)'],
    # AWPFC shorn wool production by state, Mkg greasy, columns NSW, VIC, WA, SA, TAS, QLD, AUS.
    # (FY_end, source name, PDF page, table and row, values); every value checked by eye against the rendered page,
    # 2026-10-10: all equal. PDF sha256 = the pin of the source named.
    'wool_states': ['NSW', 'VIC', 'WA', 'SA', 'TAS', 'QLD', 'AUS'],
    'wool': [
        (2017, 'awi_aug2018', 2, 'Table 2, row "2016/17 Final Estimate"', [126.0, 67.4, 71.1, 57.9, 9.2, 8.5, 340]),
        (2018, 'awi_aug2019', 3, 'Table 2, row "2017/18 Final Estimate"', [125.7, 73.5, 65.1, 59.5, 9.3, 8.3, 341]),
        (2019, 'awi_aug2020', 3, 'Table 2, row "2018/19 Final Estimate"', [99.1, 66.9, 62.2, 54.3, 9.0, 8.1, 300]),
        (2020, 'awi_aug2021', 3, 'Table 2, row "2019/20 Final Estimate"', [94.3, 63.2, 59.8, 50.0, 9.0, 7.5, 284]),
        (2021, 'awi_sep2022', 3, 'Table 2, row "2020/21"', [99.2, 70.7, 56.5, 51.5, 9.4, 7.2, 294]),
        (2022, 'awi_sep2022', 3, 'Table 2, row "2021/22"', [113.3, 73.9, 61.2, 56.1, 10.3, 8.9, 324]),
        (2023, 'awi_aug2023', 3, 'Table 2, row "2022/23"', [119.0, 70.0, 60.6, 56.7, 10.2, 11.3, 328]),
        (2024, 'awi_aug2025', 23, 'Table A1, block "2023/24", Shorn Wool Production (Mkg greasy)', [119.5, 65.6, 55.0, 58.0, 9.3, 10.1, 318.0]),
        (2025, 'awi_aug2025', 23, 'Table A1, block "2024/25", Shorn Wool Production (Mkg greasy)', [110.0, 59.5, 44.9, 47.3, 9.2, 9.2, 280.1]),
    ],
    # Wool per-head trend in the productivity input: flat (1 in every year, dry and irrigated, both sheep land uses). Basis:
    # AWPFC Aug 2024 (awi_aug2024), Table A2, greasy cut per head 4.53 kg (2003/04) and 4.44 kg (2023/24); least-squares
    # trend of ln(cut) on year FY2004-FY2024 -0.04 per cent a year. The 2010-build file declined 0.77 per cent a year (dry)
    # and 1.53 (irrigated) from 2010, source not recorded.
    'wool_trend': dict(value=1, source='awi_aug2024', where='Table A2, greasy cut per head (kg)', cut_kg={2004: 4.53, 2024: 4.44},
                       trend_pct_per_year=-0.04),
    # Dairy Australia In Focus 2025, Table 6 "Milk production by state (million litres)", PDF page 16 (printed page 14),
    # source "Dairy manufacturers"; columns NSW, VIC, QLD, SA, WA, TAS, AUS; 2024-25 flagged (p). Checked by eye against
    # the rendered page, 2026-10-10: all equal; identical to the clal.it mirror table.
    'milk_states': ['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'AUS'],
    'milk': {2017: [1141, 5732, 425, 497, 385, 836, 9016], 2018: [1144, 5979, 399, 505, 385, 913, 9325],
             2019: [1094, 5576, 359, 497, 374, 910, 8810], 2020: [1054, 5625, 315, 489, 364, 950, 8797],
             2021: [1075, 5651, 309, 500, 362, 961, 8858], 2022: [1072, 5465, 299, 490, 341, 887, 8554],
             2023: [990, 5141, 279, 472, 338, 906, 8127], 2024: [1040, 5296, 282, 479, 344, 934, 8376],
             2025: [1071, 5266, 275, 471, 331, 901, 8315]},
}
# ============================================================================= end of reference table

INTERIM = [   # item, basis, replaced by, when
    ('water_delivery_price.h5 (WP)', 'INTERIM: 2010-build values, state median on blank cells; price year unknown',
     'rebuild from the ACCC water monitoring reports', 'when the ACCC reports arrive, before the gate run'),
    ('water_licence_price.h5', 'built by water_licence_price.py (FY2020-21 public prices); INFERRED fallbacks where no public '
     'price: northern Basin proxy 2,869, median outside the Basin 1,442 $/ML', 'NSW unregulated and groundwater 71Q trades '
     '(the regulated-river table is used)', 'none planned before the gate'),
    ('live-export weight (Q3 basis)', 'carried: 0.337144 t cattle, 0.048389 t sheep per animal, 2010 basis; no public series',
     'a public per-animal live-export weight series', 'with the demand re-anchoring item on the switch list'),
    ('NT beef revenue', 'live exports only, NT meat value floored at 0', 'demand re-anchoring', 'switch list'),
    ('NT livestock cost share', 'national receipts share, unitemised receipts > 10% of broadacre receipts, cause unverified',
     'NT share once the cause is verified', 'when ABARES confirms the NT receipts breakdown'),
    ('live-export value by state', 'inferred: national FOB value per head x DAFF head by state of export port',
     'a published state live-export value', 'none planned'),
    ('drinking water WR_DRN', 'FutureBeef constants (terms unclear, not redeposited)', 'unchanged unless terms refused',
     'on the CSIRO terms ruling'),
]

ST = {1: 'NSW', 2: 'VIC', 3: 'QLD', 4: 'SA', 5: 'WA', 6: 'TAS', 7: 'NT', 8: 'ACT', 9: 'OT'}
STS = ['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT']
STATE_BY_LABEL = {'Australia': 'AUS', 'New South Wales': 'NSW', 'Victoria': 'VIC', 'Queensland': 'QLD',
                  'South Australia': 'SA', 'Western Australia': 'WA', 'Tasmania': 'TAS', 'Northern Territory': 'NT',
                  'Australian Capital Territory': 'ACT'}
FDPST = {'New South Wales': 'NSW', 'Victoria': 'VIC', 'Queensland': 'QLD', 'South Australia': 'SA', 'Western Australia': 'WA',
         'Tasmania': 'TAS', 'Northern Territory': 'NT', 'Australia': 'AUS'}
TYPES = ('BEEF', 'SHEEP', 'DAIRY')
LU_TYPE = {'BEEF': (1, 2), 'SHEEP': (14, 15), 'DAIRY': (5, 6)}   # (modified, natural) ag_landuses.csv rows
LU_NAT = [2, 6, 15, 23]
S_AWI = 'AWI Australian Wool Production Forecasting Committee (AWPFC)'
S_DA = 'Dairy Australia In Focus 2025 Table 6'


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
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (luto-livestock-economics/1.0)',
                                                       'Accept': '*/*'})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as e:          # network retry
            last = e
            time.sleep(5 * (attempt + 1))
    raise SystemExit(f'download failed from {url}: {last}')


MONTHS = {m: i for i, m in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 1)}


def daff_table(f):
    """DAFF export workbook as a table under its header row (the row holding 'Quantity'), blank rows and columns dropped."""
    d = pd.read_excel(f, header=None).dropna(how='all').dropna(how='all', axis=1)
    i = d.index[d.apply(lambda r: 'Quantity' in r.values, axis=1)][0]
    d.columns = d.loc[i]
    return d.loc[i + 1:].reset_index(drop=True)


def daff_used(name, d):
    """(financial year ending, rows and columns of `d` the tool uses): cattle and sheep rows in the window."""
    if name == 'daff_2016_2021':
        fy = pd.Series([int(y) + 1 if MONTHS[m] >= 7 else int(y) for m, y in zip(d['Month'], d['Year'])])
        keep = fy.between(2017, 2020) | ((fy == 2021) & (d.Year.astype(int) == 2020))
        cols = ['Month', 'Year', 'State', 'Species', 'Quantity']
    else:
        fy = d['Fin Year'].astype(str).str[:4].astype(int) + 1
        keep = fy.between(2021, 2025)
        cols = ['Fin Year', 'State', 'Species', 'Quantity']
    keep &= d.Species.isin(['Cattle', 'Sheep'])
    return fy[keep], d.loc[keep, cols]


def daff_content(name, path):
    """sha256 of the table's column names and of the type and value of every cell used (the DAFF content pin)."""
    d = daff_table(path)
    h = hashlib.sha256(('layout=' + repr([str(c) for c in d.columns]) + '|').encode())
    for r in daff_used(name, d)[1].itertuples(index=False):
        h.update(('|'.join(f'{type(v).__name__}:{v!r}' for v in r) + '\n').encode())
    return h.hexdigest()


def fetch_content(name, cache, offline, overrides):
    """As fetch(), for a source pinned on content (CONTENT_PINS). Cache order: the file retrieved at pinning, then any other
    cached copy of the same URL; else an override copy or a download. A copy whose content differs stops the run
    (downloads are set aside in _quarantine); a copy whose raw bytes differ but content matches is used and reported."""
    url, rpin, _ = SOURCES[name]
    cpin = CONTENT_PINS[name]
    idx = os.path.join(cache, 'index.csv')
    known = []
    if os.path.exists(idx):
        with open(idx, encoding='utf-8', newline='') as f:
            known = sorted(r['sha256'] for r in csv.DictReader(f) if r['url'] == url and r['sha256'] != rpin)
    for d in [rpin] + known:
        p = cache_path(cache, d, url)
        if os.path.exists(p):
            if daff_content(name, p) != cpin:
                raise SystemExit(f'{name}: cached file {p} does not match its content pin {cpin}; remove it and rerun')
            return p
    if name in overrides:
        with open(overrides[name], 'rb') as f:
            data = f.read()
        how = 'override ' + overrides[name]
    elif offline:
        raise SystemExit(f'--offline: {name} ({url}) is not in the cache {cache}')
    else:
        data = download(url)
        how = 'download'
    raw = sha(data)
    p = cache_path(cache, raw, url)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    part = p + '.part.xlsx'
    with open(part, 'wb') as f:
        f.write(data)
    got = daff_content(name, part)
    if got != cpin:
        q = os.path.join(cache, '_quarantine')
        os.makedirs(q, exist_ok=True)
        qp = os.path.join(q, f'{name}_{raw}.xlsx')
        os.replace(part, qp)
        raise SystemExit(f'{name}: {how} {url} has content {got}, pinned {cpin} (raw sha256 {raw}, {len(data)} bytes). '
                         f'Copy set aside at {qp}; the pin is not changed.')
    os.replace(part, p)
    cache_index_add(cache, name, url, raw, len(data))
    if raw != rpin:
        print(f'{name}: raw bytes {raw} differ from the file pinned ({rpin}); content matches ({how})', flush=True)
    return p


def fetch(name, cache, offline, overrides):
    """Path of the pinned raw file `name` in the cache; download it, or take an override copy, on a cache miss.
    Any byte difference from the pin stops the run; a changed download is set aside in _quarantine, never re-pinned."""
    if name in CONTENT_PINS:
        return fetch_content(name, cache, offline, overrides)
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


# ----------------------------------------------------------------------------- CPI
def cpi_fy(zp):
    z = zipfile.ZipFile(zp)
    b = z.read('6401018.xlsx')
    assert pd.read_excel(io.BytesIO(b), sheet_name='Data1', header=None).iloc[9, 1] == REF['cpi_series']
    d = pd.read_excel(io.BytesIO(b), sheet_name='Data1', header=None, skiprows=10)
    s = pd.Series(d[1].astype(float).values, index=pd.to_datetime(d[0]))
    fy = s.groupby(s.index.year + (s.index.month >= 7)).agg(['mean', 'size'])
    fy = fy[fy['size'] == 4]['mean'].rename('cpi_fy_mean').to_frame()
    fy['to_FY2021'] = fy.loc[REF['cpi_base_fy'], 'cpi_fy_mean'] / fy['cpi_fy_mean']
    return fy['to_FY2021']


# ----------------------------------------------------------------------------- livestock long table
def add(rows, fy, state, product, measure, value, source):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return
    rows.append(dict(FY_end=fy, state=state, product=product, measure=measure, value=value, source=source))


def abs7215(path):
    """Quarterly Original series by state; FY = sum of the four quarters Sep(Y-1)..Jun(Y)."""
    import openpyxl
    ws = openpyxl.load_workbook(path, read_only=True)['Data1']
    rs = list(ws.iter_rows(values_only=True))
    hdr, typ = rs[0], rs[2][1:]
    data = [r for r in rs[9:] if hasattr(r[0], 'year')]
    STX = {'Total (State)': 'AUS', **{k: v for k, v in STATE_BY_LABEL.items() if k != 'Australia'}}
    out = {}
    for j, h in enumerate(hdr[1:], 1):
        if typ[j - 1] != 'Original':
            continue
        st = STX[[p.strip() for p in h.split(';') if p.strip()][-1]]
        q = {}
        for r in data:
            fy = r[0].year + 1 if r[0].month >= 7 else r[0].year
            if r[j] is not None:
                q.setdefault(fy, []).append(r[j])
        out[st] = {fy: (sum(v), len(v)) for fy, v in q.items()}
    return out


def meat_rows(rows, P):
    res = {k: abs7215(P[f'abs_{n}']) for k, n in [('cattle_meat', '7215009'), ('calves_meat', '7215010'),
                                                   ('sheep_meat', '7215011'), ('lambs_meat', '7215012')]}
    for keys, prod, mname in [(['cattle_meat', 'calves_meat'], 'Beef and veal', 'meat_produced_beef_and_veal'),
                              (['sheep_meat', 'lambs_meat'], 'Sheepmeat', 'meat_produced_lamb_and_mutton')]:
        for st in ['AUS'] + STS:
            for fy in REF['fy']:
                vals = []
                for k in keys:
                    d = res[k].get(st, {}).get(fy)
                    if not d or d[1] < 4:
                        vals = None
                        break
                    vals.append(d[0])
                if vals:
                    add(rows, fy, st, prod, mname, sum(vals), 'ABS 7215.0')


def abares_rows(rows, path):
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(open(path, 'rb').read()), read_only=True)

    def ab(table, label, prod, measure):
        rs = list(wb[table].iter_rows(values_only=True))
        hdr = rs[7]
        hit = [r for r in rs[8:] if any(x is not None for x in r) and re.fullmatch(
            re.escape(label.lower()) + r'( [a-z])?', str(next(x for x in r if x is not None)).strip().lower())]
        r = hit[0]
        for j, h in enumerate(hdr):
            m = re.match(r'\s*(\d{4}).(\d\d)', str(h)) if h else None
            if not m:
                continue
            fy = int(m.group(1)) + 1
            if fy in REF['fy']:
                try:
                    v = float(r[j])
                except Exception:
                    continue
                add(rows, fy, 'AUS', prod, measure, v, 'ABARES ' + table)
    ab('Table 8', 'Cattle d', 'Live cattle exports', 'live_exports_number')
    ab('Table 8', 'Sheep e', 'Live sheep exports', 'live_exports_number')
    ab('Table 13', 'Feeder and slaughter cattle', 'Live cattle exports', 'export_value_feeder_slaughter_cattle')
    ab('Table 13', 'Breeder cattle', 'Live cattle exports', 'export_value_breeder_cattle')
    ab('Table 13', 'Live sheep', 'Live sheep exports', 'export_value')
    d = {}
    for r in rows:
        if r['measure'] in ('export_value_feeder_slaughter_cattle', 'export_value_breeder_cattle'):
            d.setdefault(r['FY_end'], []).append(r['value'])
    for fy, v in d.items():
        if len(v) == 2:
            add(rows, fy, 'AUS', 'Live cattle exports', 'export_value_total_cattle', sum(v), 'ABARES Table 13')


def daff_rows(rows, p16, p21):
    a = daff_table(p16)
    a['fy'] = [int(y) + 1 if MONTHS[m] >= 7 else int(y) for m, y in zip(a['Month'], a['Year'])]
    a = a[(a.fy.between(2017, 2020)) | ((a.fy == 2021) & (a.Year.astype(int) == 2020))].copy()
    b = daff_table(p21)
    b['fy'] = b['Fin Year'].astype(str).str[:4].astype(int) + 1
    b = b[b.fy.between(2021, 2025)].copy()
    a['file'] = 'a'
    b['file'] = 'b'
    cols = ['fy', 'State', 'Species', 'Mode', 'End Use', 'Quantity', 'file']
    c = pd.concat([a[cols], b[cols]])
    c['Quantity'] = pd.to_numeric(c['Quantity'])
    c.loc[c.fy == 2021, 'file'] = 'a+b'
    for sp, prod in [('Cattle', 'Live cattle exports'), ('Sheep', 'Live sheep exports')]:
        s = c[c.Species == sp]
        for (fy, st, fn), g in s.groupby(['fy', 'State', 'file']):
            add(rows, fy, st, prod, 'exports_number_all_modes', float(g.Quantity.sum()), 'DAFF')


def wool_milk_rows(rows):
    for fy, _, _, _, v in REF['wool']:
        for s, x in zip(REF['wool_states'], v):
            add(rows, fy, s, 'Wool', 'shorn_wool_production_greasy', x, S_AWI)
    for fy, v in REF['milk'].items():
        for s, x in zip(REF['milk_states'], v):
            add(rows, fy, s, 'Milk', 'milk_production', float(x), S_DA)


# ---- ABS 7503 / Australian Agriculture livestock values (state rows)
def clean_rc(s):
    s = str(s).strip()
    if re.fullmatch(r'\d+\.0', s):
        s = s[:-2]
    return s


def read_csv_flex(path, hdr_skip=4):
    raw = open(path, 'rb').read()
    try:
        txt = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        txt = raw.decode('cp1252')
    return pd.read_csv(io.StringIO(txt), skiprows=hdr_skip, dtype=str, keep_default_na=False)


def frame_from_rows(df):
    if df.shape[1] == 8:
        df.columns = ['rc', 'label', 'code', 'desc', 'v1', 'r1', 'v2', 'r2']
    else:
        df.columns = ['rc', 'label', 'desc', 'v1', 'r1', 'v2', 'r2'][:df.shape[1]]
        df.insert(2, 'code', '')
    return df


def read_excel_cube(path):
    x = pd.ExcelFile(path)
    out = []
    for sh in x.sheet_names:
        raw = x.parse(sh, header=None, dtype=str)
        hdr = next((i for i in range(min(12, len(raw))) if str(raw.iat[i, 0]).strip() == 'Region code'), None)
        if hdr is None:
            continue
        body = frame_from_rows(raw.iloc[hdr + 1:].copy().fillna('').iloc[:, :raw.shape[1]].copy())
        body['sheet'] = sh
        out.append(body)
    return pd.concat(out, ignore_index=True)


def normalise_7503(df, fy):
    df = df.copy()
    df['rc'] = df['rc'].map(clean_rc)
    df['desc'] = df['desc'].astype(str).str.strip()
    m = df['rc'].isin([str(i) for i in range(9)]) & df['label'].isin(STATE_BY_LABEL.keys()) & (df['desc'] != '')
    df = df[m].copy()
    df['state'] = df['label'].map(STATE_BY_LABEL)
    parts = []
    for vcol, meas in [('v1', 'gross_value_aud'), ('v2', 'local_value_aud')]:
        n = pd.to_numeric(df[vcol].astype(str).str.strip().str.replace(',', '', regex=False), errors='coerce')
        parts.append(pd.DataFrame({'FY_end': fy, 'state': df['state'], 'item_desc': df['desc'], 'measure': meas, 'value': n,
                                   'series': '7503', 'release_date': '', 'source': 'ABS 7503.0 VACP'}))
    return pd.concat(parts, ignore_index=True)


def release_of(path):
    x = pd.ExcelFile(path)
    parts = []
    for sh in x.sheet_names[:2]:
        d = x.parse(sh, header=None, nrows=8)
        parts += [str(v) for v in d.values.ravel() if isinstance(v, str)]
    m = re.search(r'Released at[^\n]*?\)\s*(?:\w{3}\s+)?(\d{1,2} \w+ \d{4})', '\n'.join(parts))
    return pd.to_datetime(m.group(1), format='%d %B %Y').strftime('%Y-%m-%d') if m else ''


def read_aa_workbook(path):
    x = pd.ExcelFile(path)
    rel = release_of(path)
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
            mu = re.search(r'\(([^()]*)\)\s*$', desc)
            if not (mu and mu.group(1).strip() == '$' and 'local value' in desc.lower()):
                continue      # only local values ($) are used here
            for j, h in ycols:
                v = r.iloc[j]
                try:
                    n = float(str(v).strip().replace(',', '')) if not isinstance(v, (int, float)) else float(v)
                except ValueError:
                    n = np.nan
                out.append((int(h[:4]) + 1, STATE_BY_LABEL[reg], desc, 'local_value_aud', n, 'AA_livestock', rel,
                            'ABS Australian Agriculture livestock value (AA_livestock)'))
    return pd.DataFrame(out, columns=['FY_end', 'state', 'item_desc', 'measure', 'value', 'series', 'release_date', 'source'])


def abs_value_rows(rows, P):
    parts = [normalise_7503(read_excel_cube(P['abs_7503_2017']), 2017)]
    for fy in (2018, 2019):
        d = read_csv_flex(P[f'abs_7503_{fy}'])
        d.columns = ['rc', 'label', 'code', 'desc', 'v1', 'r1', 'v2', 'r2']
        parts.append(normalise_7503(d, fy))
    for fy in (2020, 2021, 2022):
        parts.append(normalise_7503(read_excel_cube(P[f'abs_7503_{fy}']), fy))
    aa = pd.concat([read_aa_workbook(P[f'abs_aal_{y}']) for y in (2023, 2024, 2025)], ignore_index=True)
    key = ['FY_end', 'state', 'item_desc', 'measure', 'series']
    aa = aa[aa['release_date'] == aa.groupby(key)['release_date'].transform('max')]
    d = pd.concat(parts + [aa], ignore_index=True)
    d = d[d.FY_end.between(2017, 2025) & d.item_desc.str.contains('Cattle|Sheep|Wool|Milk', na=False)
          & ~d.item_desc.str.contains('Goats|Total|Eggs|n.e.c', na=False)]
    d = d.drop_duplicates(['FY_end', 'state', 'item_desc', 'measure', 'value'])
    for r in d.itertuples():
        p = 'Beef and veal' if 'attle' in r.item_desc else 'Sheepmeat' if 'heep' in r.item_desc else 'Wool' if 'ool' in r.item_desc else 'Milk'
        add(rows, int(r.FY_end), r.state, p, 'ABS: ' + r.item_desc + ' [' + r.measure + ']', float(r.value), r.source)


# ----------------------------------------------------------------------------- Farm Data Portal
def fdp_wide(ps, pn):
    s = pd.read_csv(ps)
    n = pd.read_csv(pn)
    n['State'] = 'Australia'
    long = pd.concat([s, n[s.columns]], ignore_index=True)
    key = ['Year', 'State', 'Industry', 'Variable']
    long = long[~long.duplicated(key, keep=False)]      # duplicated keys stay blank in the verbatim column
    return long.pivot(index=['Year', 'State', 'Industry'], columns='Variable', values='Value').reset_index() \
        .sort_values(['State', 'Industry', 'Year']).reset_index(drop=True)


def fdp_costs(f, cpi, rc):
    DSE = REF['dse']
    f = f.copy()
    f['st'] = f.State.map(FDPST)
    f = f[f.Year.between(2017, 2025)].copy()

    def g(r, k):
        v = r.get(k, np.nan)
        return 0.0 if pd.isna(v) else float(v)
    QC_COMMON, AC, FLC, FOC, FDC = (REF[k] for k in ('QC_COMMON', 'AC', 'FLC', 'FOC', 'FDC'))
    rows = []
    for _, r in f.iterrows():
        c = cpi[r.Year]
        tot = g(r, 'Total cash receipts ($)')
        crop = g(r, 'Total crop gross receipts ($)')
        lsh = 1 - crop / tot if tot > 0 else np.nan
        if r.Industry == 'All Broadacre':
            beef = g(r, 'Beef herd at 30 June (no.)')
            sheep = g(r, 'Sheep flock at 30 June (no.)')
            dse_b, dse_s = beef * DSE['BEEF'], sheep * DSE['SHEEP']
            tot_dse = dse_b + dse_s
            if tot_dse <= 0:
                continue
            mod = g(r, 'Beef cattle receipts ($)') + g(r, 'Sheep receipts ($)') + g(r, 'Wool receipts ($)')
            unitem = tot - crop - mod - g(r, 'Other livestock receipts ($)') - g(r, 'Other farm income ($)')
            for t, head, dse in [('BEEF', beef, dse_b), ('SHEEP', sheep, dse_s)]:
                sh = dse / tot_dse
                out = dict(st=r.st, Year=r.Year, type=t, head_june=head, mod_receipts=mod * c, lv_receipts=(tot - crop) * c,
                           tot_receipts=tot * c, unitem_receipts=unitem * c,
                           QC=(sum(g(r, k) for k in QC_COMMON) * sh + (g(r, 'Shearing crutching ($)') if t == 'SHEEP' else 0.0)) * c,
                           AC=sum(g(r, k) for k in AC) * lsh * sh * c, FLC=sum(g(r, k) for k in FLC) * lsh * sh * c,
                           FOC=sum(g(r, k) for k in FOC) * lsh * sh * c, FDC=sum(g(r, k) for k in FDC) * lsh * sh * c)
                if t == 'BEEF':
                    out['net_receipts'] = (g(r, 'Beef cattle receipts ($)') - g(r, 'Beef cattle purchased ($)')) * c
                else:
                    out['net_receipts'] = (g(r, 'Sheep receipts ($)') - g(r, 'Sheep purchased ($)')) * c
                    out['wool_receipts'] = g(r, 'Wool receipts ($)') * c
                    out['wool_kg'] = g(r, 'Wool produced (kg)')
                rows.append(out)
        elif r.Industry == 'Dairy':
            head = g(r, 'Dairy cattle at June 30 (no.)')
            if head <= 0:
                continue
            rows.append(dict(st=r.st, Year=r.Year, type='DAIRY', head_june=head, mod_receipts=g(r, 'Milk receipts excluding freight ($)') * c,
                             lv_receipts=(tot - crop) * c,
                             QC=(sum(g(r, k) for k in QC_COMMON) + g(r, 'Dairy supplies ($)')) * lsh * c,
                             AC=sum(g(r, k) for k in AC) * lsh * c, FLC=sum(g(r, k) for k in FLC) * lsh * c,
                             FOC=sum(g(r, k) for k in FOC) * lsh * c, FDC=sum(g(r, k) for k in FDC) * lsh * c,
                             milk_receipts=g(r, 'Milk receipts excluding freight ($)') * c, milk_l=g(r, 'Milk produced (ltrs)')))
    D = pd.DataFrame(rows)
    cols = [c for c in ['QC', 'AC', 'FLC', 'FOC', 'FDC', 'net_receipts', 'wool_receipts', 'wool_kg', 'milk_receipts', 'milk_l'] if c in D.columns]
    S = D.groupby(['st', 'type']).apply(lambda q: pd.Series(
        {c: q[c].sum() / q.head_june.sum() for c in cols} | {'years': q.Year.nunique(),
         'receipts_share': min(q.mod_receipts.sum() / q.lv_receipts.sum(), 1.0) if q.lv_receipts.sum() > 0 else np.nan}))
    rc = rc.copy()
    rc['st'] = rc.state_name.str.upper()
    rc['type'] = rc.type.str.upper()
    ratio = rc.set_index(['st', 'type']).nir_over_abs.replace(0, np.nan)
    natr = rc.groupby('type').apply(lambda q: q.nir_state_total.sum() / q.abs_state_total.sum())
    S['nir_over_abs'] = [ratio.get((s, t), np.nan) if s != 'AUS' else natr[t] for s, t in S.index]
    S['nir_over_abs'] = S.nir_over_abs.fillna(pd.Series([natr[t] for _, t in S.index], index=S.index))
    for c in cols:
        if c not in ('wool_kg', 'milk_l'):
            S[c + '_per_nir_head'] = S[c] / S.nir_over_abs
    ba = D[D.type == 'BEEF'].groupby('st')[['unitem_receipts', 'tot_receipts']].sum()
    ush = ba.unitem_receipts / ba.tot_receipts
    S['unitemised_share_broadacre'] = [ush.get(s, np.nan) for s, _ in S.index]
    S['receipts_share_own'] = S.receipts_share
    S['share_basis'] = 'state'
    for s, t in [(s, t) for s, t in S.index if s != 'AUS' and ush.get(s, 0) > REF['unitemised_max']]:
        S.loc[(s, t), 'receipts_share'] = S.loc[('AUS', t), 'receipts_share']
        S.loc[(s, t), 'share_basis'] = 'national: unitemised receipts > 10% (cause unverified)'
    for c in ['QC', 'AC', 'FLC', 'FOC', 'FDC']:
        S[c + '_full_per_nir_head'] = S[c + '_per_nir_head']
        S[c + '_per_nir_head'] = S[c + '_full_per_nir_head'] * S.receipts_share
    return S


# ----------------------------------------------------------------------------- ABS 4618 grazed-pasture water
def water_state_volume(P):
    STN = {k: v for k, v in STATE_BY_LABEL.items() if k != 'Australia'}
    rows = []
    for fy in range(REF['fy_water'][0], REF['fy_water'][1] + 1):
        p = P[f'abs_4618_{fy}']
        if fy in (2018, 2019):
            raw = pd.read_csv(p, header=None, encoding='latin1', dtype=str)
            frames = [raw]
        else:
            x = pd.ExcelFile(p)
            frames = [pd.read_excel(x, sheet_name=s, header=None, dtype=object) for s in x.sheet_names
                      if s not in ('Contents', 'Guide to estimates')]
        for raw in frames:
            hi = next(i for i in range(min(30, len(raw))) if str(raw.iloc[i, 0]).strip() == 'Region code')
            d = raw.iloc[hi + 1:].copy()
            d.columns = [str(c).strip() for c in raw.iloc[hi]]
            d = d.rename(columns={'Region code': 'code', 'Region label': 'label', 'Estimate': 'est'})
            d = d.rename(columns={c: 'item' for c in d.columns if c.startswith('Commodity description')})
            d = d[d['item'].notna() & d['label'].notna()]
            for code, label, item, est in zip(d['code'], d['label'], d['item'], d['est']):
                try:
                    code = str(int(float(str(code).strip())))
                except ValueError:
                    code = str(code).strip()
                if len(code) != 1 or code == '0':
                    continue
                item = str(item).strip()
                m = re.match(r'^(?:Water use - )?(?P<grp>.+?) - Volume applied \(ML\)(?: \([a-z]\))?$', item)
                if not m or not m.group('grp').startswith('Pastures (including lucerne) and cereal crops used for grazing'):
                    continue
                if isinstance(est, (int, float, np.integer, np.floating)):
                    v = float(est)
                else:
                    try:
                        v = float(str(est).strip().replace(',', ''))
                    except ValueError:
                        v = np.nan
                rows.append((fy, code, item, str(label).strip(), v))
    x = pd.DataFrame(rows, columns=['FY_end', 'code', 'item', 'region_name', 'value']).drop_duplicates(['FY_end', 'code', 'item'])
    x = x.assign(st=x.region_name.map(STN).fillna(x.region_name))
    return x.groupby(['st', 'FY_end']).value.sum().groupby('st').mean()


def livestock_water(V, state, ty, lm, head):
    DSE = REF['dse']
    TC = {0: 'BEEF', 1: 'SHEEP', 2: 'DAIRY'}
    wr_drn = {t: v * 365 / 1e6 for t, v in REF['drink_l_head_day'].items()}
    irr = (lm == 1) & (ty >= 0)
    H = pd.DataFrame({'st': state[irr], 't': pd.Series(ty[irr]).map(TC).values, 'h': head[irr]}).groupby(['st', 't']).h.sum().unstack().fillna(0)
    dse_sum = (H * pd.Series(DSE)).sum(axis=1)
    natV, natD = V.sum(), dse_sum.sum()
    lo, hi = REF['water_band']
    rows = []
    for s in STS:
        v = float(V.get(s, 0.0))
        d = float(dse_sum.get(s, 0.0))
        nat = natV / natD
        raw = v / d if d > 0 and v > 0 else nat
        per_dse = float(np.clip(raw, nat * lo, nat * hi))
        for t in DSE:
            rows.append(dict(state=s, type=t, V_ml=v, irr_head=float(H.loc[s, t]) if s in H.index and t in H.columns else 0.0,
                             WR_IRR_ml_head=per_dse * DSE[t], WR_DRN_ml_head=wr_drn[t],
                             src=('state' if d > 0 and v > 0 else 'national') + ('; capped' if abs(per_dse - raw) > 1e-12 else '')))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- base-year head on cells
def fix_state_ties(sa2, STid, mask_path):
    """As livestock_map.py: a cell whose SA2 state digit disagrees with state_id takes the SA2 of the nearest
    cell (grid distance) whose SA2 digit equals that state_id and agrees with its own state_id."""
    from scipy.spatial import cKDTree
    with rasterio.open(mask_path) as r:
        mk = r.read(1) == 1
    rows, cols = np.nonzero(mk)
    dig = sa2 // 10**8
    bad = np.where(dig != STid)[0]
    out = sa2.copy()
    for s in np.unique(STid[bad]):
        good = np.where((dig == s) & (STid == s))[0]
        b = bad[STid[bad] == s]
        if len(good) == 0:
            continue
        tree = cKDTree(np.c_[rows[good], cols[good]])
        _, j = tree.query(np.c_[rows[b], cols[b]])
        out[b] = sa2[good[j]]
    return out


def base_head(lu, lm, A, PK, SN, SM, sa2, recon_sa2, ratings, kcell):
    """Pasture head per cell of the livestock map: k(SA2, type) x capacity / (DSE x grassfed), the livestock_map.py
    formula with its full-precision ratings and SA2 k (livestock_recon_sa2.csv)."""
    ty = np.full(lu.shape, -1, np.int8)
    for i, t in enumerate(TYPES):
        ty[np.isin(lu, LU_TYPE[t])] = i
    cap = np.nan_to_num(PK * np.where(np.isin(lu, LU_NAT), SN, SM) * A * np.where(lm == 1, 2., 1.) / 365.)
    head = np.zeros(len(lu))
    for i, t in enumerate(TYPES):
        r = recon_sa2[recon_sa2.type == t.lower()].set_index('sa2_code').k
        m = ty == i
        k = r.reindex(sa2[m]).to_numpy()
        ok = np.isfinite(k)
        assert (np.float32(k[ok]) == kcell[t][m][ok]).all(), f'{t}: SA2 k disagrees with lvstk_k.h5'
        h = np.zeros(m.sum())
        h[ok] = k[ok] * cap[m][ok] / (ratings[t] * REF['grassfed'][t])
        head[m] = h
    return ty, head


# ----------------------------------------------------------------------------- outputs per head and prices
def outputs_and_prices(L, cpi, NIR):
    FY = REF['fy']
    LE_W = REF['live_export_t_per_head']

    def ser(product, measure, state='AUS', src=None):
        x = L[(L['product'] == product) & (L.measure == measure) & (L.state == state) & L.FY_end.isin(FY)]
        if src:
            x = x[x.source.str.contains(src, regex=False)]
        return x.groupby('FY_end').value.sum()

    def defl(s):
        return (s * s.index.map(cpi)).mean() if len(s) else np.nan

    def local_value(product, vacp_measure, aa_measure, state):
        a = ser(product, vacp_measure, state)
        b = ser(product, aa_measure, state)
        return defl(pd.concat([a[a.index <= 2022], b[b.index >= 2023]]))
    V = {}
    for st in STS + ['AUS']:
        V[('cattle', st)] = local_value('Beef and veal', 'ABS: Livestock slaughtered and other disposals - Cattle and calves [local_value_aud]',
                                        'ABS: Cattle disposals - Local value ($) [local_value_aud]', st)
        V[('sheep', st)] = local_value('Sheepmeat', 'ABS: Livestock slaughtered and other disposals - Sheep and lambs [local_value_aud]',
                                       'ABS: Sheep disposals - Local value ($) [local_value_aud]', st)
        wv = [m for m in L[L['product'] == 'Wool'].measure.unique() if 'local_value_aud' in m and '7503' not in m]
        a = pd.concat([ser('Wool', m, st) for m in wv if 'Livestock' in m])
        a = a[a.index <= 2022].groupby(level=0).max()
        b = ser('Wool', 'ABS: Wool - Local value ($) [local_value_aud]', st)
        V[('wool', st)] = defl(pd.concat([a, b[b.index >= 2023]]))
        V[('milk', st)] = local_value('Milk', 'ABS: Livestock products - Milk [local_value_aud]', 'ABS: Milk - Local value ($) [local_value_aud]', st)

    def wmean(s, deflate=False):
        return (s * s.index.map(cpi)).mean() if deflate else s.mean()
    Q_beef = wmean(ser('Beef and veal', 'meat_produced_beef_and_veal'))
    Q_sm = wmean(ser('Sheepmeat', 'meat_produced_lamb_and_mutton'))
    N_lc = wmean(ser('Live cattle exports', 'live_exports_number')) * 1e3
    V_lc = wmean(ser('Live cattle exports', 'export_value_total_cattle') * 1e6, True)
    N_ls = wmean(ser('Live sheep exports', 'live_exports_number')) * 1e3
    V_ls = wmean(ser('Live sheep exports', 'export_value') * 1e6, True)
    Q_wool = wmean(ser('Wool', 'shorn_wool_production_greasy', src=S_AWI)) * 1e6 / 1e3      # Mkg -> t
    Q_milk = wmean(ser('Milk', 'milk_production', src=S_DA)) * 1e6                         # ML -> L
    P = {'BEEF P1': (V[('cattle', 'AUS')] - V_lc) / Q_beef, 'BEEF P3': V_lc / (N_lc * LE_W['BEEF']),
         'SHEEP P1': (V[('sheep', 'AUS')] - V_ls) / Q_sm, 'SHEEP P2': V[('wool', 'AUS')] / Q_wool,
         'SHEEP P3': V_ls / (N_ls * LE_W['SHEEP']), 'DAIRY P1': V[('milk', 'AUS')] / Q_milk}

    def dshare(product):
        x = L[(L['product'] == product) & (L.measure == 'exports_number_all_modes') & (L.state != 'AUS') & L.FY_end.isin(FY)]
        s = x.groupby('state').value.sum()
        return s / s.sum()
    le_sh = {'BEEF': dshare('Live cattle exports'), 'SHEEP': dshare('Live sheep exports')}
    wool_st = L[(L['product'] == 'Wool') & (L.measure == 'shorn_wool_production_greasy') & (L.state != 'AUS') & L.FY_end.isin(FY)
                & L.source.str.contains(S_AWI, regex=False)].groupby('state').value.mean()
    milk_st = L[(L['product'] == 'Milk') & (L.measure == 'milk_production') & (L.state != 'AUS') & L.FY_end.isin(FY)
                & L.source.str.contains(S_DA, regex=False)].groupby('state').value.mean()
    out = []
    for t, Vkey, lexp_N, lexp_V in [('BEEF', 'cattle', N_lc, V_lc), ('SHEEP', 'sheep', N_ls, V_ls)]:
        meat_v_nat = V[(Vkey, 'AUS')] - lexp_V
        vpn = lexp_V / lexp_N
        raw = {}
        for s in STS:
            le_n = lexp_N * le_sh[t].get(s, 0.0)
            raw[s] = max(V.get((Vkey, s), 0.0) - le_n * vpn, 0.0) if not np.isnan(V.get((Vkey, s), np.nan)) else 0.0
        k = meat_v_nat / sum(raw.values())
        for s in STS:
            h = NIR.get((s, t), 0.0)
            le_n = lexp_N * le_sh[t].get(s, 0.0)
            row = dict(state=s, type=t, nir_head=h, meat_t=raw[s] * k / P[f'{t} P1'], lexp_t=le_n * LE_W[t])
            if t == 'SHEEP':
                row['wool_t'] = Q_wool * wool_st.get(s, 0.0) / wool_st.sum()
            out.append(row)
    for s in STS:
        out.append(dict(state=s, type='DAIRY', nir_head=NIR.get((s, 'DAIRY'), 0.0), milk_l=Q_milk * milk_st.get(s, 0.0) / milk_st.sum()))
    O = pd.DataFrame(out)
    O['Q1'] = np.where(O.type == 'DAIRY', O.milk_l, O.meat_t) / O.nir_head
    O['Q2'] = O.get('wool_t') / O.nir_head
    O['Q3'] = O.lexp_t / O.nir_head
    nat = {t: {q: (O[O.type == t][{'Q1': 'milk_l' if t == 'DAIRY' else 'meat_t', 'Q2': 'wool_t', 'Q3': 'lexp_t'}[q]].sum()
                   / O[O.type == t].nir_head.sum()) for q in ['Q1', 'Q2', 'Q3']} for t in TYPES}
    for q in ['Q1', 'Q2', 'Q3']:
        bad = ~np.isfinite(O[q].astype(float)) | (O.nir_head <= 0)
        O.loc[bad, q] = [nat[t][q] for t in O.loc[bad, 'type']]
    win = dict(Q_beef_t=Q_beef, Q_sheepmeat_t=Q_sm, live_cattle_n=N_lc, live_cattle_value=V_lc, live_sheep_n=N_ls,
               live_sheep_value=V_ls, wool_greasy_t=Q_wool, milk_l=Q_milk, **{f'value_{k[0]}': v for k, v in V.items() if k[1] == 'AUS'})
    return P, O, nat, win


def cells(cols, state, O, nat, P, c1, c2, ty, head_ha, kk, past, sp):
    N = len(state)
    TC = {'BEEF': 0, 'SHEEP': 1, 'DAIRY': 2}
    DSE, GF = REF['dse'], REF['grassfed']

    def by_state(fn):
        return pd.Series(state).map({s: fn(s) for s in STS + ['OT']}).values.astype(float)

    def c1v(s, t, col):
        ok = (s, t) in c1.index and np.isfinite(c1.loc[(s, t), col])
        return c1.loc[(s, t), col] if ok else c1.loc[('AUS', t), col]
    data = {}
    for (f, t) in cols:
        if f in ('F1', 'F2', 'F3'):
            v = np.ones(N)
        elif f in ('Q1', 'Q2', 'Q3'):
            q = O[O.type == t].set_index('state')[f]
            v = by_state(lambda s: q.get(s, nat[t][f]))
        elif f in ('P1', 'P2', 'P3'):
            v = np.full(N, P[f'{t} {f}'])
        elif f == 'QC':
            v = by_state(lambda s: c1v(s, t, 'QC_per_nir_head'))
        elif f in ('AC', 'FLC', 'FOC', 'FDC'):
            rate = by_state(lambda s: c1v(s, t, f + '_per_nir_head'))
            pot = kk[t].values.astype(float) * past * sp / (365 * DSE[t] * GF[t])
            hh = np.where((ty == TC[t]) & (head_ha > 0), head_ha, pot)
            v = rate * np.nan_to_num(hh, nan=0.0)
        elif f == 'WR_DRN':
            v = np.full(N, c2.loc[('NSW', t), 'WR_DRN_ml_head'])
        elif f == 'WR_IRR':
            v = by_state(lambda s: c2.loc[(s, t), 'WR_IRR_ml_head'] if (s, t) in c2.index else c2.WR_IRR_ml_head.xs(t, level='type').median())
        else:
            raise SystemExit(f'agec_lvstk.h5 column {(f, t)} has no rule')
        data[(f, t)] = v.astype(np.float32)
    A = pd.DataFrame(data)[cols]
    A.index = pd.RangeIndex(N)
    return A


def receipts_report(A, WP, c1, state, area, lm, ty, head):
    """Receipts share report: margins per head on cells of the type with pasture head, FDP reconciliation."""
    K = ['QC', 'AC', 'FLC', 'FOC', 'FDC']
    rows = []
    for c, t in enumerate(TYPES):
        on = (ty == c) & (head > 0)
        for s in STS + ['AUS']:
            k = on & ((state == s) if s != 'AUS' else True)
            h = head[k]
            if h.sum() == 0:
                continue
            a = lambda n: A[(n, t)].values[k].astype(float)
            rev = sum(a('F%d' % i) * a('Q%d' % i) * a('P%d' % i) for i in (1, 2, 3) if ('F%d' % i, t) in A.columns)
            wr = a('WR_DRN') + np.where(lm[k] == 1, a('WR_IRR'), 0)
            cost = (h * a('QC')).sum() + (area[k] * (a('AC') + a('FLC') + a('FOC') + a('FDC'))).sum()
            r = dict(type=t, state=s, head_on_cells=h.sum(), revenue_per_head=(h * rev).sum() / h.sum(), cost_per_head=cost / h.sum(),
                     water_per_head=(h * wr * WP[k]).sum() / h.sum())
            r['margin_per_head'] = r['revenue_per_head'] - r['cost_per_head'] - r['water_per_head']
            key = (s, t) if (s, t) in c1.index and np.isfinite(c1.loc[(s, t), 'receipts_share']) and c1.loc[(s, t), 'QC_full_per_nir_head'] > 0 else ('AUS', t)
            r['fdp_row_used'] = key[0]
            r['receipts_share'] = c1.loc[key, 'receipts_share']
            r['share_basis'] = c1.loc[key, 'share_basis']
            r['fdp_full_cost_per_nir_head'] = sum(c1.loc[key, x + '_full_per_nir_head'] for x in K)
            r['fdp_charged_cost_per_nir_head'] = sum(c1.loc[key, x + '_per_nir_head'] for x in K)
            r['share_uncharged'] = 1 - r['receipts_share']
            r['map_cost_over_fdp_charged'] = r['cost_per_head'] / r['fdp_charged_cost_per_nir_head']
            r['map_cost_over_fdp_full'] = r['cost_per_head'] / r['fdp_full_cost_per_nir_head']
            rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- provenance helpers
def flat_wool(raw):
    """yieldincreases_bau2022.csv with every wool column (dry and irrigated; WOOL_COLS) set to the reference trend value in
    every year row. A text edit of those fields only: every other byte of the file is kept (two header rows: lm, product)."""
    nl = b'\r\n' if b'\r\n' in raw else b'\n'
    lines = raw.split(nl)
    lm, pr = lines[0].decode().split(','), lines[1].decode().split(',')
    assert len(lm) == len(pr)
    wcol = [i for i, p in enumerate(pr) if p.strip().endswith('WOOL')]
    assert len(wcol) == 4 and all(pr[i].strip() in WOOL_COLS for i in wcol), [pr[i] for i in wcol]
    out, nrow = [lines[0], lines[1]], 0
    for ln in lines[2:]:
        if not ln.strip():
            out.append(ln)
            continue
        f = ln.decode().split(',')
        assert len(f) == len(pr)
        for i in wcol:
            f[i] = str(REF['wool_trend']['value'])
        out.append(','.join(f).encode())
        nrow += 1
    return nl.join(out), [(lm[i], pr[i]) for i in wcol], nrow


def content_sha(a):
    a = np.ascontiguousarray(a)
    return sha(f'{a.dtype.str}|{a.shape}|'.encode() + a.tobytes())


def pkg_versions():
    out = {}
    for p in ('numpy', 'pandas', 'tables', 'openpyxl', 'xlrd', 'rasterio', 'scipy', 'xarray', 'netCDF4'):
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--input', required=True, help='folder of the inherited 2010-build files listed in INHERITED (input/Nick_build/inherited_2010)')
    ap.add_argument('--livestock-map', required=True, help='output folder of livestock_map.py')
    ap.add_argument('--pasture', required=True, help='pasture_kg_dm_ha.h5 from pasture_productivity.py')
    ap.add_argument('--sa2', required=True, help='sa2_2021_cells.npy from nlum_map.py')
    ap.add_argument('--out', required=True, help='output folder (e.g. input/Nick_build/livestock_economics)')
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
    P = {k: fetch(k, a.cache, a.offline, overrides) for k in SOURCES}
    inp = {f: os.path.join(a.input, f) for f in INHERITED}
    LMF = ['lumap.h5', 'lmmap.h5', 'lvstk_k.h5', 'livestock_recon_state_type.csv', 'livestock_recon_sa2.csv', 'livestock_dse_ratings.csv']
    upstream = {f: os.path.join(a.livestock_map, f) for f in LMF}
    upstream['pasture_kg_dm_ha.h5'] = a.pasture
    upstream['sa2_2021_cells.npy'] = a.sa2
    in_sha = {f: sha_file(p) for f, p in {**inp, **upstream}.items()}
    if in_sha['yieldincreases_bau2022.csv'] != YIELDINC_SHA:
        raise SystemExit(f"{inp['yieldincreases_bau2022.csv']} has sha256 {in_sha['yieldincreases_bau2022.csv']}, not the 2010-build "
                         f'file ({YIELDINC_SHA}); refusing to run')

    # ---- tables
    cpi = cpi_fy(P['abs_cpi'])
    rows = []
    meat_rows(rows, P)
    abares_rows(rows, P['abares_agcomm'])
    daff_rows(rows, P['daff_2016_2021'], P['daff_2021_2026'])
    wool_milk_rows(rows)
    abs_value_rows(rows, P)
    L = pd.DataFrame(rows)
    rc = pd.read_csv(upstream['livestock_recon_state_type.csv'])
    rc_up = rc.assign(st=rc.state_name.str.upper(), t=rc.type.str.upper())
    NIR = rc_up.set_index(['st', 't']).nir_state_total
    Pr, O, nat, win = outputs_and_prices(L, cpi, NIR)
    fdp = fdp_wide(P['fdp_state'], P['fdp_national'])
    c1 = fdp_costs(fdp, cpi, rc)
    V = water_state_volume(P)
    print('tables done', f'{time.time() - t0:.0f}s', flush=True)

    # ---- cells
    STid = np.load(inp['state_id.npy'])
    state = pd.Series(STid).map(ST).values
    A_ha = pd.read_hdf(inp['real_area.h5']).to_numpy().astype(np.float64)
    SN = pd.read_hdf(inp['safe_pur_natl.h5']).to_numpy().astype(np.float64)
    SM = pd.read_hdf(inp['safe_pur_modl.h5']).to_numpy().astype(np.float64)
    past = pd.read_hdf(a.pasture).to_numpy().astype(np.float64)
    lu = pd.read_hdf(upstream['lumap.h5']).to_numpy().astype(np.int64)
    lm = pd.read_hdf(upstream['lmmap.h5']).to_numpy().astype(np.int64)
    kk = pd.read_hdf(upstream['lvstk_k.h5'])
    rat = pd.read_csv(upstream['livestock_dse_ratings.csv']).set_index('animal').intake_kg_dm_day.to_dict()
    sa2 = fix_state_ties(np.load(a.sa2).astype(np.int64), STid.astype(np.int64), inp['NLUM_2010-11_mask.tif'])
    assert len(lu) == len(past) == len(sa2) == len(state) == NCELL
    ty, head = base_head(lu, lm, A_ha, past, SN, SM, sa2, pd.read_csv(upstream['livestock_recon_sa2.csv']), rat,
                         {t: kk[t].to_numpy() for t in TYPES})
    with np.errstate(divide='ignore', invalid='ignore'):
        head_ha = head / A_ha
    c2 = livestock_water(V, state, ty, lm, head)
    cols = list(pd.read_hdf(inp['agec_lvstk.h5'], stop=1).columns)
    AG = cells(cols, state, O, nat, Pr, c1.reset_index().set_index(['st', 'type']), c2.set_index(['state', 'type']),
               ty, head_ha, kk, past, SN)
    wdp = pd.read_hdf(inp['water_delivery_price.h5'])
    med = wdp.groupby(state).median()
    fill = wdp.copy()
    m = fill.isna()
    fill[m] = pd.Series(state[m.values]).map(med).values
    fill = fill.fillna(float(wdp.median())).astype(wdp.dtype)
    rep = receipts_report(AG, fill.to_numpy().astype(float).ravel(), c1.reset_index().set_index(['st', 'type']), state, A_ha, lm, ty, head)
    print('cells done', f'{time.time() - t0:.0f}s', flush=True)

    # ---- write (layout of the 2010-base files)
    outs = {'agec_lvstk.h5': ('agec_lvstk', AG), 'water_delivery_price.h5': ('water_delivery_price', fill)}
    for f, (key, obj) in outs.items():
        obj.to_hdf(os.path.join(a.out, f), key=key, mode='w', format='table', index=False, complevel=9, complib='zlib')
    ncs = [NC.write_like(a.out, 'agec_lvstk', AG, ['field', 'animal']), NC.write_like(a.out, 'water_delivery_price', fill)]
    pd.DataFrame([dict(item=k, value=v) for k, v in {**{f'price {k}': v for k, v in Pr.items()}, **win}.items()]) \
        .to_csv(os.path.join(a.out, 'livestock_prices.csv'), index=False, lineterminator='\n')
    O.to_csv(os.path.join(a.out, 'livestock_outputs_per_head.csv'), index=False, lineterminator='\n')
    c1.to_csv(os.path.join(a.out, 'livestock_costs_per_head.csv'), lineterminator='\n')
    c2.to_csv(os.path.join(a.out, 'livestock_water_per_head.csv'), index=False, lineterminator='\n')
    rep.to_csv(os.path.join(a.out, 'receipts_share_report.csv'), index=False, lineterminator='\n')
    write_csv(os.path.join(a.out, 'interim_items_livestock_economics.csv'), ['item', 'basis', 'replaced_by', 'when'], INTERIM)
    # ---- productivity input with a flat wool per-head trend (drop-in for the 2010-build file)
    yi, wcols, nrow = flat_wool(open(inp['yieldincreases_bau2022.csv'], 'rb').read())
    open(os.path.join(a.out, 'yieldincreases_bau2022.csv'), 'wb').write(yi)
    print('written', f'wool columns {wcols}, {nrow} year rows set flat;', f'{time.time() - t0:.0f}s', flush=True)

    # ---- provenance (last)
    me = open(os.path.abspath(__file__), 'rb').read()
    PV = [['source', k, url, pin, 'sha256 of raw file', note] for k, (url, pin, note) in SOURCES.items()
          if k not in CONTENT_PINS]
    for k in CONTENT_PINS:
        url, rpin, note = SOURCES[k]
        PV += [['source', k, url, CONTENT_PINS[k], 'sha256 of the rows and columns used (enforced)', note],
               ['source_raw', k, url, sha_file(P[k]), f'sha256 of the raw file used (recorded; pinned raw file {rpin})', note]]
    PV += [['inherited', f, '', in_sha[f], 'inherited 2010-build input (inherited_2010)', ''] for f in INHERITED]
    PV += [['upstream', f, '', in_sha[f], 'input from another 2021-base tool', upstream[f].replace('\\', '/')] for f in upstream]
    PV += [['script', 'input/Nick_build/scripts/livestock_economics.py', '', sha(me), 'sha256 of script', '']]
    PV += [['package', k, '', v, 'version', ''] for k, v in sorted(pkg_versions().items())]
    PV += [['python', sys.version.split()[0], '', '', 'interpreter', '']]
    for f, (key, _) in outs.items():
        p = os.path.join(a.out, f)
        back = pd.read_hdf(p)
        PV.append(['output', f, '', sha_file(p), 'sha256 of file (HDF5 bytes vary between runs)', ''])
        if isinstance(back, pd.DataFrame):
            for c in back.columns:
                PV.append(['dataset', f'{f}:/{key} values {c}', '', content_sha(back[c].to_numpy()), 'sha256 of dtype, shape and array bytes', ''])
        cl = list(back.columns) if isinstance(back, pd.DataFrame) else [back.name]
        PV.append(['dataset', f'{f}:/{key} values', '', content_sha(back.to_numpy()), 'sha256 of dtype, shape and array bytes',
                   f'{len(cl)} columns {back.to_numpy().dtype} {back.shape}'])
        PV.append(['dataset', f'{f}:/{key} index', '', content_sha(back.index.to_numpy()), 'sha256 of dtype, shape and array bytes', 'cell order'])
    for f in ('livestock_prices.csv', 'livestock_outputs_per_head.csv', 'livestock_costs_per_head.csv', 'livestock_water_per_head.csv',
              'receipts_share_report.csv', 'interim_items_livestock_economics.csv', 'yieldincreases_bau2022.csv'):
        PV.append(['output', f, '', sha_file(os.path.join(a.out, f)), 'sha256',
                   'flat wool per-head trend (a change to default behaviour)' if f.startswith('yieldincreases') else ''])
    PV += [['netcdf', os.path.basename(p), '', h, 'sha256 of NetCDF content (nc_layout.content_sha)', 'checked equal to the .h5 arrays'] for p, h in ncs]
    write_csv(os.path.join(a.out, 'provenance.csv'), ['kind', 'item', 'url', 'sha256', 'note', 'source'], PV)
    print('done', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
