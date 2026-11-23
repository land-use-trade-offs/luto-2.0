#!/usr/bin/env python
"""Build LUTO's water licence price layer (water_licence_price.h5): FY2020-21 water entitlement prices, FY2021 dollars.

Sources (public, downloaded at run time, sha256-checked against pinned values; fetched and pinned only, the values used are
hand-checked entries of the reference table below, each with its page):
  - Ricardo (formerly Aither, now WSP) Water Markets Report 2020-21 review and 2021-22 outlook: Table 4 (p.27, southern
    Murray-Darling Basin entitlement VWAP by trading zone and class, from the Victorian, NSW and SA water registers) and the
    northern Basin general security VWAPs in the text (p.33, NSW 71Q licence-to-licence transfers). No licence stated.
  - Queensland Permanent Water Trading Annual Report, surface water, supplemented supply, 1 Jul 2020 to 30 Jun 2021
    (Transfer of Ownership - Water Allocations: weighted average price by water supply scheme and priority group). The
    department host refuses scripted clients; the file is the Internet Archive copy of the department's URL.
  - Tasmanian Irrigation scheme offers and charge sheets: primary-sale (list) prices for new irrigation rights, not
    secondary-market trades: Scottsdale (2020-21 charge sheet), Sassafras Wesley Vale Augmentation (Apr 2022), Greater
    Meander (May 2022, floor price). No licence stated.
  - NSW Water Trade dashboard, 71Q entitlement trade table (crosstab "71Q Data (Table)", regulated rivers, registration
    dates June 2020 to October 2026; one row per trade: water source, licence category, price per share, shares). The
    dashboard is a Tableau view with no scripted download: the file was downloaded by hand on 2026-10-19 and is supplied
    with --source nsw_71q=PATH on the first run, then read from the cache. Parsed by the tool (FY2020-21 VWAP by water
    source and category); the values used are also entered below and must match. CC BY 3.0 AU (data.gov.au record).
  - ABS 6401.0 Consumer Price Index, Aug 2026 (All groups CPI, Australia, A2325846C): FY2021 dollars for the Tasmanian
    prices dated outside FY2021 (financial-year mean of four quarters).
Inherited 2010-build inputs from --input (inherited_2010/, copies of 2010-build files), each sha256 recorded in the provenance file: rivreg_id.h5,
rivreg_lut.h5, draindiv_id.h5, draindiv_lut.h5, REGION_STATE_r.h5, real_area.h5 and water_licence_price.h5 (the interim
layer, pinned: its key, index, dtype and series name are the output layout, and its values are the comparison only).

Method:
  1. Price: the FY2020-21 entitlement (permanent) price in each state's main class: NSW general security, Victorian high
     reliability, SA high security, Queensland supplemented medium priority. A single base year (FY2020-21), unlike the
     FY2017-25 window of the crop and livestock economics. Nominal FY2020-21 prices are FY2021 dollars; the Tasmanian
     list prices dated in FY2020 or FY2022 are converted by the CPI.
  2. Spatial key: river region (BoM Geofabric, rivreg_id.h5) by state (REGION_STATE_r.h5). Each (river region, state)
     pair in the reference table takes its zone or scheme price on every cell; the table names, for each pair, whether
     the zone is the one the source reports (STATED) or a neighbouring or containing zone (INFERRED zone).
  3. Pairs not in the table (no public price): in the northern Murray-Darling Basin river regions, the northern proxy
     2,869 $/ML (mean of NSW Gwydir and Macquarie general security, 3,782 and 1,955); everywhere else, including southern
     Basin river regions without a public price, the median of the public prices outside the Basin in FY2021 dollars,
     computed from the zone prices below and checked against the ruled 1,442 $/ML. Both are labelled INFERRED.
  4. Integer $/ML (round half up), int16, the layout of the 2010-build file (key /water_licence_price, table format, the
     series name WATER_PRICE_ML_ABARES kept so the file is a drop-in; the values are not ABARES').
Outputs in --out: water_licence_price.h5 and water_licence_price.nc (NetCDF copy in the jinzhu layout, one variable named
as the series, nc_layout.py); water_licence_price_zones.csv (every (river region, state) pair: cells, area,
irrigated area on the 2021-base map, rule, zone, price); water_licence_price_compare.csv (the layer against the interim by
drainage division and state, cell-area and irrigated-area weighted); water_licence_price_nsw71q.csv (FY2020-21 VWAP by NSW
water source and licence category from the 71Q table, with the zone each feeds or is compared with);
provenance.csv (last).

Run as a plain script from the repository root (no luto imports):

    python input/Nick_build/scripts/water_licence_price.py --input input/Nick_build/inherited_2010 --map-dir input/Nick_build/livestock_map --out input/Nick_build/water_licence_price
        --cache input/Nick_build/_cache [--offline] [--source NAME=PATH]

Dependencies: numpy, pandas, PyTables, openpyxl, xarray, netCDF4. No luto imports (nc_layout.py is a sibling module).
Output values are deterministic; HDF5 file bytes carry a group timestamp, so provenance records dataset content hashes.
"""
import argparse
import csv
import hashlib
import io
import os
import sys
import time
import urllib.request
import zipfile
from importlib import metadata

import numpy as np
import pandas as pd

import nc_layout as NC

# ----------------------------------------------------------------------------- sources and pins
QLD = 'https://www.rdmw.qld.gov.au/__data/assets/pdf_file/0006/1590045/pwtr-supplemented-annual-2020-21.pdf'
TI = 'https://www.tasmanianirrigation.com.au/source-assets/'
SOURCES = {   # name: (url, sha256 of the raw file, note)
    'abs_cpi': ('https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/consumer-price-index-australia/'
                'aug-2026/CPI-Quarterly.zip', '0b97b11375f6cdbf87b59a85d9c4cd990950e43d09aeea4504221f72d9af3cbd',
                'ABS 6401.0 CPI Aug 2026, quarterly zip (6401018.xlsx), CC BY 4.0'),
    'ricardo_2021': ('https://www.ricardo.com/media/z5aplhxw/2021-ricardo-water-markets-report.pdf',
                     'd5a8d37b12c6542d72916260f35d7f4d9b3502f0d929cf410702b024420e3fb4',
                     'Ricardo Water Markets Report 2020-21 review and 2021-22 outlook (3,204,560 B); no licence stated'),
    'qld_pwtr_supp_2021': ('https://web.archive.org/web/2023id_/' + QLD,
                           '9c4694749c49b3c29db841dff7080d2d42d6e81ababafbecdf531894a162e83d',
                           'Qld Permanent Water Trading Annual Report, surface water, supplemented, FY2020-21 (182,400 B); '
                           'Internet Archive copy of ' + QLD + ' (host refuses scripted clients); (c) State of Queensland'),
    'ti_sis_2021': (TI + 'map-data/scottsdale/downloads/Annual-Charges-2020-21_SIS.pdf',
                    '76c9186a3ee14cf3ff435531e36c5885162e4c389f34d8ccbe03f9f37819677e',
                    'Tasmanian Irrigation, Scottsdale Irrigation Scheme charge rates 2020-21 (786,406 B)'),
    'ti_swisa_offer': ('https://tasmanianirrigation.com.au/source-assets/map-data/SWISA-Water-Sales-One-Page-Summary.pdf',
                       '4d0c06af7d830c7d6347dc2721504464377629f7c3bff9565acf5a444dc04fb5',
                       'Tasmanian Irrigation, Sassafras Wesley Vale Augmentation water sales, April 2022 (103,505 B)'),
    'ti_gmis_offer': ('https://tasmanianirrigation.com.au/source-assets/map-data/3.-GMIS-Water-Sales-Scheme-Offer.pdf',
                      'a80da713faac30bc64daba550282f8c6a72af77a1da044e2cb562c9e74b542a8',
                      'Tasmanian Irrigation, Greater Meander water sales scheme offer, May 2022 (304,022 B)'),
    'nsw_71q': ('https://tableau.dpie.nsw.gov.au/t/DPIEExternal/views/WaterTradeDashboard/WaterTradeSnapshot',
                '24e667212b1ff038ab77dfd779d677f939d2980b74670dea5241f57f99ac647b',
                'NSW Water Trade dashboard, Download > Crosstab "71Q Data (Table)", UTF-16 tab-separated, 411,370 B, '
                'downloaded by hand 2026-10-19 (no scripted download; supply with --source); CC BY 3.0 AU'),
}
INHERITED = ['rivreg_id.h5', 'rivreg_lut.h5', 'draindiv_id.h5', 'draindiv_lut.h5', 'REGION_STATE_r.h5', 'real_area.h5',
             'water_licence_price.h5']
INTERIM_SHA = '9d6957768c5f164230ea4ef7e34dcfc38ba47b92e96752728619a68809a0d6c9'   # the 2010-build layer (layout, comparison)
MAPF = ['lumap.h5', 'lmmap.h5']
NCELL = 6956407

# ============================================================================= reference table (hand-entered)
# Each price with its source, PDF page and where on the page; every value checked against the PDF text (pdftotext -raw and
# -layout), 2026-10-19: all equal. PDF sha256 = the pin of the source named. Changing any value is a ruling.
R27 = 'Table 4, annual VWAP 2020-21'
REF = {
    'cpi_series': 'A2325846C', 'cpi_base_fy': 2021,
    # zone: (nominal $/ML, financial year of the price, source, PDF page, where, note)
    'price': {
        'vic_1a_hrws': (4084, 2021, 'ricardo_2021', 27, R27 + ', row "Vic 1A Greater Goulburn HRWS"', '632 trades, 39,108 ML'),
        'vic_6_hrws': (4466, 2021, 'ricardo_2021', 27, R27 + ', row "Vic 6 Murray (Dart to Barmah) HRWS"', '130 trades, 13,315 ML'),
        'vic_7_hrws': (6157, 2021, 'ricardo_2021', 27, R27 + ', row "Vic 7 Murray (Barmah to SA) HRWS"', '623 trades, 28,858 ML'),
        'nsw_murray_gs': (1870, 2021, 'ricardo_2021', 27, R27 + ', row "NSW Murray GS" (zones 10 and 11)', '70 trades, 28,301 ML'),
        'nsw_mbidgee_gs': (2029, 2021, 'ricardo_2021', 27, R27 + ', row "NSW Murrumbidgee GS"', '41 trades, 74,282 ML'),
        'sa_murray_hs': (6815, 2021, 'ricardo_2021', 27, R27 + ', row "SA Murray HS"', '131 trades, 9,712 ML'),
        'nsw_lachlan_gs': (1220, 2021, 'ricardo_2021', 33, 'text: "NSW Lachlan GS increased 82 per cent from $671 per ML to '
                           '$1,220 per ML" (2017-18 to 2020-21, 71Q transfers)', ''),
        'nsw_macquarie_gs': (1955, 2021, 'ricardo_2021', 33, 'text: "NSW Macquarie and Cudgegong GS increased 55 per cent from '
                             '$1,258 per ML to $1,955 per ML"', ''),
        'nsw_gwydir_gs': (3782, 2021, 'ricardo_2021', 33, 'text: "NSW Gwydir GS increased 89 per cent from $2,000 per ML to '
                          '$3,782 per ML"', ''),
        'qld_mareeba_mp': (3775, 2021, 'qld_pwtr_supp_2021', 1, 'Mareeba Dimbulah WSS, Medium', '77 trades, 2,412 ML'),
        'qld_burdekin_mp': (364, 2021, 'qld_pwtr_supp_2021', 1, 'Burdekin Haughton WSS, Medium', '23 trades, 25,466 ML'),
        'qld_bundaberg_mp': (1390, 2021, 'qld_pwtr_supp_2021', 1, 'Bundaberg WSS, Medium', '100 trades, 8,464 ML'),
        'qld_nogoa_mp': (2154, 2021, 'qld_pwtr_supp_2021', 1, 'Nogoa Mackenzie WSS, Medium', '38 trades, 9,043 ML'),
        'qld_logan_mp': (1047, 2021, 'qld_pwtr_supp_2021', 1, 'Logan River WSS, Medium', '18 trades, 1,432 ML'),
        'qld_macintyre_mp': (2550, 2021, 'qld_pwtr_supp_2021', 1, 'Macintyre Brook WSS, Medium', '1 trade, 92 ML; Border Rivers '
                             'WSS Medium 8 trades, 1,866 ML at $0 (nil consideration), not used'),
        'qld_central_brisbane_mp': (2549, 2021, 'qld_pwtr_supp_2021', 2, 'Central Brisbane River WSS, Medium', '7 trades, 650 ML'),
        'qld_mary_valley_mp': (832, 2021, 'qld_pwtr_supp_2021', 2, 'Mary Valley WSS, Medium', '15 trades, 1,113 ML'),
        'ti_scottsdale': (1400, 2021, 'ti_sis_2021', 1, '"Water Entitlement Pricing 2020-21 ($/ML)", Scottsdale, all areas, '
                          '$1,400.00', 'primary-sale list price (unchanged from the 2015 round two offer)'),
        'ti_swisa': (2090, 2022, 'ti_swisa_offer', 1, 'offer price "$2,090 / ML" (Sassafras Wesley Vale Augmentation, 9,200 ML; '
                     'April 2022, offer closed 17 June 2022)', 'primary-sale list price'),
        'ti_gmis': (1550, 2022, 'ti_gmis_offer', 4, 'offer price "$1,550 / ML" (Greater Meander, 11,000 ML; release May 2022, '
                    'offer closed 24 June 2022; FAQ: the lowest price at which water is sold)', 'primary-sale floor price'),
    },
    # Qld Condamine-Balonne (river region 42, QLD): volume-weighted mean of the scheme Medium rows in the same table, p.1
    # (Chinchilla Weir 1 trade, 30 ML, 4,500; St George 7 trades, 514 ML, 3,529; Upper Condamine 4 trades, 554 ML,
    # 3,378). Upper Condamine Risk Class A (2 trades, 915 ML at $0) is not medium priority and not used.
    'qld_condamine_parts': [(4500, 30), (3529, 514), (3378, 554)],
    # NSW 71Q table (nsw_71q): FY2020-21 = registration dates 1 Jul 2020 to 30 Jun 2021; VWAP = sum(price per share x
    # shares) / sum(shares). zone: (water source, licence category, expected VWAP rounded, trades, shares); the tool
    # recomputes each and stops on any difference. Only zones on a fallback before the 71Q table are added; the NSW Basin
    # zones keep the Ricardo values (the 71Q FY2020-21 VWAPs are reported alongside in water_licence_price_nsw71q.csv).
    # No FY2020-21 trades for Lower or Upper Namoi, NSW Border Rivers, Barwon-Darling, Richmond or Paterson.
    'nsw71q': {
        'nsw_hunter_gs': ('HUNTER REGULATED RIVER WATER SOURCE', 'REGULATED RIVER (GENERAL SECURITY)', 1973, 2, 110),
        'nsw_bega_gs': ('BEGA AND BROGO REGULATED RIVERS WATER SOURCE', 'REGULATED RIVER (GENERAL SECURITY)', 1000, 1, 66),
    },
    # 71Q sources compared with the Ricardo zone values (same class), reported only
    'nsw71q_compare': {
        'nsw_murray_gs': 'NEW SOUTH WALES MURRAY REGULATED RIVER WATER SOURCE',
        'nsw_mbidgee_gs': 'MURRUMBIDGEE REGULATED RIVER WATER SOURCE',
        'nsw_lachlan_gs': 'LACHLAN REGULATED RIVER WATER SOURCE',
        'nsw_macquarie_gs': 'MACQUARIE AND CUDGEGONG REGULATED RIVERS WATER SOURCE',
        'nsw_gwydir_gs': 'GWYDIR REGULATED RIVER WATER SOURCE',
    },
    # (river region id, state): (zone, label). STATED: the zone the source reports covers the river region;
    # INFERRED zone: the river region is supplied from, or lies within, the zone named (no separate public price).
    'pairs': {
        (142, 'NSW'): ('nsw_murray_gs', 'STATED'), (15, 'NSW'): ('nsw_murray_gs', 'INFERRED zone'),
        (201, 'NSW'): ('nsw_murray_gs', 'INFERRED zone'), (202, 'NSW'): ('nsw_murray_gs', 'INFERRED zone'),
        (119, 'NSW'): ('nsw_murray_gs', 'INFERRED zone'),
        (143, 'NSW'): ('nsw_mbidgee_gs', 'STATED'), (14, 'NSW'): ('nsw_mbidgee_gs', 'INFERRED zone'),
        (109, 'NSW'): ('nsw_lachlan_gs', 'STATED'),
        (122, 'NSW'): ('nsw_macquarie_gs', 'STATED'), (37, 'NSW'): ('nsw_macquarie_gs', 'INFERRED zone'),
        (84, 'NSW'): ('nsw_gwydir_gs', 'STATED'),
        (80, 'VIC'): ('vic_1a_hrws', 'STATED'), (117, 'VIC'): ('vic_1a_hrws', 'INFERRED zone'),
        (22, 'VIC'): ('vic_1a_hrws', 'INFERRED zone'), (34, 'VIC'): ('vic_1a_hrws', 'INFERRED zone'),
        (142, 'VIC'): ('vic_6_hrws', 'STATED'), (157, 'VIC'): ('vic_6_hrws', 'INFERRED zone'),
        (202, 'VIC'): ('vic_6_hrws', 'INFERRED zone'), (102, 'VIC'): ('vic_6_hrws', 'INFERRED zone'),
        (201, 'VIC'): ('vic_7_hrws', 'STATED'), (119, 'VIC'): ('vic_7_hrws', 'INFERRED zone'),
        (119, 'SA'): ('sa_murray_hs', 'STATED'), (120, 'SA'): ('sa_murray_hs', 'STATED'),
        (42, 'QLD'): ('qld_condamine_mp', 'INFERRED zone (volume-weighted mean of three schemes)'),
        (19, 'QLD'): ('qld_macintyre_mp', 'INFERRED zone (Macintyre Brook WSS; single trade, 92 ML)'),
        (92, 'NSW'): ('nsw_hunter_gs', 'STATED (Hunter regulated river; 2 trades, 110 shares)'),
        (12, 'NSW'): ('nsw_bega_gs', 'STATED (Bega and Brogo regulated rivers; single trade, 66 shares)'),
        (65, 'QLD'): ('qld_nogoa_mp', 'STATED'),
        (107, 'QLD'): ('qld_bundaberg_mp', 'STATED'), (29, 'QLD'): ('qld_bundaberg_mp', 'INFERRED zone'),
        (30, 'QLD'): ('qld_bundaberg_mp', 'INFERRED zone'),
        (28, 'QLD'): ('qld_burdekin_mp', 'STATED'), (86, 'QLD'): ('qld_burdekin_mp', 'STATED'),
        (9, 'QLD'): ('qld_mareeba_mp', 'STATED'), (130, 'QLD'): ('qld_mareeba_mp', 'INFERRED zone'),
        (21, 'QLD'): ('qld_central_brisbane_mp', 'STATED'), (118, 'QLD'): ('qld_logan_mp', 'STATED'),
        (127, 'QLD'): ('qld_mary_valley_mp', 'STATED'),
        (193, 'TAS'): ('ti_gmis', 'STATED list price'), (162, 'TAS'): ('ti_scottsdale', 'STATED list price'),
        (128, 'TAS'): ('ti_swisa', 'STATED list price'), (173, 'TAS'): ('ti_swisa', 'INFERRED zone'),
    },
    # Fallbacks. Northern Basin river regions: Border Rivers, Castlereagh, Condamine-Culgoa, Darling,
    # Gwydir, Macquarie-Bogan, Moonie, Namoi, Paroo, Warrego.
    'basin': 'Murray-Darling Basin',
    'north_rivreg': [19, 37, 42, 47, 84, 122, 132, 146, 158, 207],
    'north_proxy': 2869,          # (3782 + 1955) / 2 = 2868.5, ruled as 2,869
    # Median of the public zone prices outside the Basin, FY2021 dollars (integer $/ML): Qld coastal schemes, Tasmanian
    # Irrigation, NSW coastal regulated rivers. With the ten Qld and Tasmanian prices (364, 832, 1047, 1390, 1400, 1484, 2000,
    # 2154, 2549, 3775) and with the two NSW coastal prices added (1000, 1973) the median is (1400 + 1484) / 2 = 1442.
    'nonbasin_zones': ['qld_mareeba_mp', 'qld_burdekin_mp', 'qld_bundaberg_mp', 'qld_nogoa_mp', 'qld_logan_mp',
                       'qld_central_brisbane_mp', 'qld_mary_valley_mp', 'ti_scottsdale', 'ti_swisa', 'ti_gmis',
                       'nsw_hunter_gs', 'nsw_bega_gs'],
    'nonbasin_median': 1442,      # ruled; the computed median must equal it
}
# ============================================================================= end of reference table

ST = {'New South Wales': 'NSW', 'Victoria': 'VIC', 'Queensland': 'QLD', 'South Australia': 'SA', 'Western Australia': 'WA',
      'Tasmania': 'TAS', 'Northern Territory': 'NT', 'Australian Capital Territory': 'ACT', 'Other Territories': 'OT'}


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
    ext = os.path.splitext(url.split('?')[0].rsplit('/', 1)[-1])[1]
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
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (luto-water-licence-price/1.0)', 'Accept': '*/*'})
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


# ----------------------------------------------------------------------------- CPI
def cpi_fy(zp):
    z = zipfile.ZipFile(zp)
    b = z.read('6401018.xlsx')
    assert pd.read_excel(io.BytesIO(b), sheet_name='Data1', header=None).iloc[9, 1] == REF['cpi_series']
    d = pd.read_excel(io.BytesIO(b), sheet_name='Data1', header=None, skiprows=10)
    s = pd.Series(d[1].astype(float).values, index=pd.to_datetime(d[0]))
    fy = s.groupby(s.index.year + (s.index.month >= 7)).agg(['mean', 'size'])
    fy = fy[fy['size'] == 4]['mean']
    return fy.loc[REF['cpi_base_fy']] / fy


def rhu(x):
    """Round half up to an integer."""
    return int(np.floor(x + 0.5))


# ----------------------------------------------------------------------------- NSW 71Q table
def nsw71q_fy2021(path):
    """FY2020-21 VWAP by (water source, licence category) from the 71Q crosstab (UTF-16, tab-separated, one row per trade)."""
    d = pd.read_csv(path, sep='\t', encoding='utf-16', dtype=str)
    cols = ['Registration Date', 'Water Source Type', 'Water Source', 'WMZ', 'Licence Category', 'Price Per Share',
            'Shares Traded']
    assert list(d.columns) == cols, list(d.columns)
    d['date'] = pd.to_datetime(d['Registration Date'], format='%d/%m/%Y')
    for c in ('Price Per Share', 'Shares Traded'):
        d[c] = d[c].str.replace(',', '', regex=False).astype(float)
    d = d[(d.date >= '2020-07-01') & (d.date <= '2021-06-30')]
    d['value'] = d['Price Per Share'] * d['Shares Traded']
    g = d.groupby(['Water Source', 'Licence Category']).agg(trades=('value', 'size'), shares=('Shares Traded', 'sum'),
                                                            value=('value', 'sum'))
    g['vwap'] = g.value / g.shares
    return g.drop(columns='value').reset_index()


# ----------------------------------------------------------------------------- zone prices, FY2021 dollars
def zone_prices(cpi, N):
    Z = {}
    for z, (p, fy, src, page, where, note) in REF['price'].items():
        f = float(cpi.loc[fy])
        Z[z] = dict(price=rhu(p * f), nominal=p, fy=fy, to_FY2021=f, source=src, page=page, where=where, note=note)
    parts = REF['qld_condamine_parts']
    v = sum(p * ml for p, ml in parts) / sum(ml for _, ml in parts)
    Z['qld_condamine_mp'] = dict(price=rhu(v), nominal=v, fy=2021, to_FY2021=1.0, source='qld_pwtr_supp_2021', page=1,
                                 where='Condamine and Balonne: Chinchilla Weir, St George, Upper Condamine WSS, Medium',
                                 note='volume-weighted mean ' + ', '.join(f'{p} x {ml} ML' for p, ml in parts))
    for z, (ws, cat, want, n, sh) in REF['nsw71q'].items():
        r = N[(N['Water Source'] == ws) & (N['Licence Category'] == cat)]
        assert len(r) == 1, (z, len(r))
        r = r.iloc[0]
        if (rhu(r.vwap), int(r.trades), int(r.shares)) != (want, n, sh):
            raise SystemExit(f'{z}: 71Q FY2020-21 gives {r.vwap:.1f} $/ML, {r.trades} trades, {r.shares} shares; '
                             f'reference table {want}, {n}, {sh}')
        Z[z] = dict(price=rhu(r.vwap), nominal=float(r.vwap), fy=2021, to_FY2021=1.0, source='nsw_71q', page='',
                    where=f'{ws}, {cat}, registration dates 1 Jul 2020 to 30 Jun 2021, VWAP',
                    note=f'{n} trade{"s" if n > 1 else ""}, {sh} shares')
    med = float(np.median([Z[z]['price'] for z in REF['nonbasin_zones']]))
    if rhu(med) != REF['nonbasin_median']:
        raise SystemExit(f'median of the public prices outside the Basin is {med}, ruled {REF["nonbasin_median"]}')
    return Z, med


def content_sha(a):
    a = np.ascontiguousarray(a)
    return sha(f'{a.dtype.str}|{a.shape}|'.encode() + a.tobytes())


def pkg_versions():
    out = {}
    for p in ('numpy', 'pandas', 'tables', 'openpyxl', 'xarray', 'netCDF4'):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def write_csv(path, header, rows):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)


def compare(D, key):
    """The layer against the interim by `key`: cell-area-weighted over all cells, irrigated-area-weighted on the
    2021-base irrigated cells, and the irrigated share at $0 and by rule."""
    rows = []
    for k, x in list(D.groupby(key)) + [('Australia', D)]:
        irr = x.irr
        ih = x.ha[irr].sum()
        r = dict(area=k, area_kha=x.ha.sum() / 1e3, irrigated_kha=ih / 1e3,
                 interim_mean_all=np.average(x.interim, weights=x.ha), layer_mean_all=np.average(x.price, weights=x.ha),
                 interim_mean_irrigated=np.average(x.interim[irr], weights=x.ha[irr]) if ih > 0 else np.nan,
                 layer_mean_irrigated=np.average(x.price[irr], weights=x.ha[irr]) if ih > 0 else np.nan,
                 interim_zero_share_irrigated=x.ha[irr & (x.interim == 0)].sum() / ih if ih > 0 else np.nan)
        for rule in ('table', 'north_proxy', 'nonbasin_median'):
            r[f'irrigated_share_{rule}'] = x.ha[irr & (x.rule == rule)].sum() / ih if ih > 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows).round(4)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--input', required=True, help='folder of the inherited 2010-build files listed in INHERITED (input/Nick_build/inherited_2010)')
    ap.add_argument('--map-dir', required=True, help='2021-base map folder (lumap.h5, lmmap.h5): the comparison weights only')
    ap.add_argument('--out', required=True, help='output folder (e.g. input/Nick_build/water_licence_price)')
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
    mp = {f: os.path.join(a.map_dir, f) for f in MAPF}
    in_sha = {f: sha_file(p) for f, p in {**inp, **mp}.items()}
    if in_sha['water_licence_price.h5'] != INTERIM_SHA:
        raise SystemExit(f"{inp['water_licence_price.h5']} has sha256 {in_sha['water_licence_price.h5']}, not the 2010-build "
                         f'layer ({INTERIM_SHA}); refusing to run')

    # ---- prices by zone, FY2021 dollars
    cpi = cpi_fy(P['abs_cpi'])
    N = nsw71q_fy2021(P['nsw_71q'])
    Z, med = zone_prices(cpi, N)
    print('median of the public prices outside the Basin (FY2021 $/ML):', med,
          sorted(Z[z]['price'] for z in REF['nonbasin_zones']), flush=True)
    assert {z for z, _ in REF['pairs'].values()} <= set(Z), {z for z, _ in REF['pairs'].values()} - set(Z)

    # ---- cells
    interim = pd.read_hdf(inp['water_licence_price.h5'])
    assert isinstance(interim, pd.Series) and len(interim) == NCELL, (type(interim), len(interim))
    rr = pd.Series(pd.read_hdf(inp['rivreg_id.h5']).to_numpy().ravel()).fillna(-1).astype(np.int64).to_numpy()
    rrn = pd.read_hdf(inp['rivreg_lut.h5']).set_index('HR_RIVREG_ID').HR_RIVREG_NAME
    ddn = pd.read_hdf(inp['draindiv_lut.h5']).set_index('HR_DRAINDIV_ID').HR_DRAINDIV_NAME
    dd = pd.Series(pd.read_hdf(inp['draindiv_id.h5']).to_numpy().ravel()).map(ddn).fillna('(none)').to_numpy()
    st = pd.read_hdf(inp['REGION_STATE_r.h5'])['STE_NAME11'].map(ST).fillna('(none)').to_numpy()
    ha = pd.read_hdf(inp['real_area.h5']).to_numpy().astype(np.float64).ravel()
    lu = pd.read_hdf(mp['lumap.h5']).to_numpy().ravel()
    lm = pd.read_hdf(mp['lmmap.h5']).to_numpy().ravel()
    assert len(rr) == len(dd) == len(st) == len(ha) == len(lu) == len(lm) == NCELL
    assert REF['basin'] in set(ddn.values), REF['basin']
    D = pd.DataFrame({'rr': rr, 'state': st, 'dd': dd, 'ha': ha, 'irr': (lm == 1) & (lu >= 0),
                      'interim': interim.to_numpy().astype(np.float64)})

    # ---- price per (river region, state) pair, then per cell
    G = D.groupby(['rr', 'state']).agg(cells=('ha', 'size'), area_kha=('ha', lambda x: x.sum() / 1e3),
                                       dd=('dd', lambda x: x.mode().iat[0]),
                                       basin_share=('dd', lambda x: float((x == REF['basin']).mean())))
    G['irrigated_kha'] = D[D.irr].groupby(['rr', 'state']).ha.sum().reindex(G.index).fillna(0) / 1e3
    north = set(REF['north_rivreg'])
    rows = []
    for (r, s), g in G.iterrows():
        if (r, s) in REF['pairs']:
            z, lab = REF['pairs'][(r, s)]
            rows.append(dict(rule='table', zone=z, label=lab, price=Z[z]['price']))
        elif r in north and g.basin_share > 0.5:
            rows.append(dict(rule='north_proxy', zone='', label='INFERRED (northern proxy)', price=REF['north_proxy']))
        else:
            rows.append(dict(rule='nonbasin_median', zone='', label='INFERRED (public median outside the Basin)'
                             + ('; Basin river region without a public price' if g.basin_share > 0.5 else ''),
                             price=REF['nonbasin_median']))
    G = pd.concat([G, pd.DataFrame(rows, index=G.index)], axis=1)
    missing = set(REF['pairs']) - set(G.index)
    assert not missing, f'reference pairs with no cells: {missing}'
    G.insert(0, 'rivreg_name', [rrn.get(r, '') for r, _ in G.index])
    m = D.set_index(['rr', 'state']).index
    D['price'] = G.price.reindex(m).to_numpy().astype(np.float64)
    D['rule'] = G.rule.reindex(m).to_numpy()
    assert not np.isnan(D.price).any() and (D.price > 0).all()
    out = pd.Series(D.price.to_numpy().astype(np.int16), index=interim.index, name=interim.name)
    assert (out.to_numpy() == D.price.to_numpy()).all(), 'int16 overflow'
    print('zones (FY2021 $/ML):', {z: v['price'] for z, v in Z.items()}, flush=True)
    print('cells by rule:', D.rule.value_counts().to_dict(), flush=True)

    # ---- write (layout of the 2010-build file)
    f_h5 = os.path.join(a.out, 'water_licence_price.h5')
    out.to_hdf(f_h5, key='water_licence_price', mode='w', format='table', index=False, complevel=9, complib='zlib')
    ncs = [NC.write_like(a.out, 'water_licence_price', out)]                        # NetCDF copy, jinzhu layout
    zr = G.reset_index()
    zr['zone_source'] = [f"{Z[z]['source']}{' p.' + str(Z[z]['page']) if Z[z]['page'] else ''}: {Z[z]['where']}" if z else ''
                         for z in zr.zone]
    zr['zone_nominal'] = [Z[z]['nominal'] if z else np.nan for z in zr.zone]
    zr['zone_fy'] = [Z[z]['fy'] if z else np.nan for z in zr.zone]
    zr['zone_note'] = [Z[z]['note'] if z else '' for z in zr.zone]
    zr = zr.sort_values(['state', 'irrigated_kha'], ascending=[True, False])
    zr.to_csv(os.path.join(a.out, 'water_licence_price_zones.csv'), index=False, lineterminator='\n', float_format='%.4f')
    C = pd.concat([compare(D, 'dd').assign(by='drainage division'), compare(D, 'state').assign(by='state')])
    C.to_csv(os.path.join(a.out, 'water_licence_price_compare.csv'), index=False, lineterminator='\n')
    used = {(ws, cat): z for z, (ws, cat, *_) in REF['nsw71q'].items()}
    cmp_ = {ws: z for z, ws in REF['nsw71q_compare'].items()}
    N['zone'] = [used.get((w, c), cmp_.get(w, '') if c == 'REGULATED RIVER (GENERAL SECURITY)' else '')
                 for w, c in zip(N['Water Source'], N['Licence Category'])]
    N['use'] = ['price used' if (w, c) in used else 'compared with Ricardo' if z else ''
                for w, c, z in zip(N['Water Source'], N['Licence Category'], N.zone)]
    N['zone_price'] = [Z[z]['price'] if z else np.nan for z in N.zone]
    N.round(1).to_csv(os.path.join(a.out, 'water_licence_price_nsw71q.csv'), index=False, lineterminator='\n')
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(N.round(1).to_string(index=False), flush=True)
    print(C.to_string(index=False), flush=True)

    # ---- provenance (last)
    me = open(os.path.abspath(__file__), 'rb').read()
    PV = [['source', k, url, pin, 'sha256 of raw file', note] for k, (url, pin, note) in SOURCES.items()]
    PV += [['reference', z, '', '', f"{v['price']} $/ML FY2021 (nominal {v['nominal']:.1f}, FY{v['fy']}, x {v['to_FY2021']:.6f})",
            f"{v['source']}{' p.' + str(v['page']) if v['page'] else ''}: {v['where']}" + (f"; {v['note']}" if v['note'] else '')]
           for z, v in Z.items()]
    PV += [['reference', 'north_proxy', '', '', f"{REF['north_proxy']} $/ML", 'mean of Gwydir and Macquarie general security'],
           ['reference', 'nonbasin_median', '', '', f"{REF['nonbasin_median']} $/ML",
            'median of ' + ', '.join(str(p) for p in sorted(Z[z]['price'] for z in REF['nonbasin_zones'])) + f' = {med}']]
    PV += [['inherited', f, '', in_sha[f], 'inherited 2010-build input (inherited_2010)', ''] for f in INHERITED]
    PV += [['upstream', f, '', in_sha[f], 'input from another 2021-base tool (comparison weights only)',
            mp[f].replace('\\', '/')] for f in MAPF]
    PV += [['script', 'input/Nick_build/scripts/water_licence_price.py', '', sha(me), 'sha256 of script', '']]
    PV += [['package', k, '', v, 'version', ''] for k, v in sorted(pkg_versions().items())]
    PV += [['python', sys.version.split()[0], '', '', 'interpreter', '']]
    back = pd.read_hdf(f_h5)
    PV += [['output', 'water_licence_price.h5', '', sha_file(f_h5), 'sha256 of file (HDF5 bytes vary between runs)', ''],
           ['dataset', 'water_licence_price.h5:/water_licence_price values', '', content_sha(back.to_numpy()),
            'sha256 of dtype, shape and array bytes', f'{back.name} {back.dtype} {back.shape}'],
           ['dataset', 'water_licence_price.h5:/water_licence_price index', '', content_sha(back.index.to_numpy()),
            'sha256 of dtype, shape and array bytes', 'cell order']]
    for f in ('water_licence_price_zones.csv', 'water_licence_price_compare.csv', 'water_licence_price_nsw71q.csv'):
        PV.append(['output', f, '', sha_file(os.path.join(a.out, f)), 'sha256', ''])
    PV += [['netcdf', os.path.basename(p), '', h, 'sha256 of NetCDF content (nc_layout.content_sha)', 'checked equal to the .h5 arrays'] for p, h in ncs]
    write_csv(os.path.join(a.out, 'provenance.csv'), ['kind', 'item', 'url', 'sha256', 'note', 'source'], PV)
    print('done', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
