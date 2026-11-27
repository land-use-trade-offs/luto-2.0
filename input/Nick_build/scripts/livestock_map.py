#!/usr/bin/env python
"""Build LUTO's 2021-base land-use map with livestock typing, the stubble rate and the per-type stocking calibration k.

Sources (public, downloaded at run time, sha256-checked against pinned values):
  - ABS 7121.0 Agricultural Commodities, Australia, 2020-21, ASGS Edition 3 data cube (SA2, state, national livestock).
  - DCCEEW National Inventory Report 2024 activity table, agriculture livestock (national and state populations by class).
  - Australia's National Greenhouse Accounts API (NIR 2024): national enteric CH4 for dairy cattle, pasture beef cattle and
    the six sheep classes, FY2020-21. The API is a changing source: the raw responses are cached and pinned one by one.
  - ABS ASGS Edition 3 SA2 2021 boundaries (GDA94): Greater Capital City flags for the reconciliation report.
  - ABARES NLUM v7.1 250 m ALUM v8 2020-21 (the file nlum_map.py pins): feedlot area (ALUM 5.2.2) by SA2.
Inputs from the other 2021-base tools: the NLUM v7.1 class allocation (nlum_map.py) and the pasture layer
(pasture_productivity.py); with --hir-cells, the existing-HIR-project cells of
ag_mgt_caps.py (2010 map, input/Nick_build/ag_mgt_caps) (hir_project_cells.csv, hir_scaling.csv). Inherited 2010-build inputs from --input (inherited_2010/, copies of 2010-build files), each sha256 recorded in the
provenance file: real_area.h5, safe_pur_natl.h5, safe_pur_modl.h5, state_id.npy, NLUM_2010-11_mask.tif, and lumap.h5 /
lmmap.h5 as the file layout (dtype, name, index) of the outputs of the same name. The outputs are therefore not built from
public sources alone.

Method:
  1. Feed ratings (DSE per head = kg DM/head/day): national FY2020-21 intake from NIR enteric CH4 per head, inverted with
     Charmley et al. (2015) for cattle and Howden et al. for sheep (flock-weighted over the six classes).
  2. SA2 target head per type = ABS 2020-21 SA2 head x (NIR state total / ABS state total); the typing (step 4) uses
     these. For calibrating k (step 5), beef is scaled from pasture head: the NIR beef total is pasture cattle only, so
     each SA2's ABS meat cattle first lose an estimate of feedlot head (estimated: NIR state feedlot head x the SA2's
     share of the state's NLUM 2020-21 feedlot area), and the ABS state total loses the head subtracted. The subtraction
     stops at the SA2's ABS head, and, in an SA2 with typed beef cells, where k on them would fall below the lower edge
     of the band, so no typed beef cell is left without stock.
  3. Pasture capacity per cell = pasture growth x safe utilisation (natural or modified land) x area x (2 if irrigated) / 365;
     sheep capacity of an SA2 also counts cereal stubble (stubble rate x cereal area).
  4. Typing of grazing cells in each SA2: persistent cells keep their type; other grazing cells take the type furthest below
     its DSE share (dairy first on irrigated, then highest-capacity, modified cells); natural unallocated cells enter
     where k exceeds three times the national median; persistent cells of a target-0 type are retyped or left unallocated.
  5. Town SA2s: an (SA2, type) pair above the national band (three times the national median k) in an SA2 under 20,000 ha
     is calibrated with its SA3: it takes the SA3 pool's k for the type (SA3 target over capacity), and the head its cells
     cannot carry at that k moves to the SA3's other SA2s of the type in proportion to their target (the SA3 total is
     unchanged). ABS counts a holding's stock at its location, and an SA2 that small is a settlement.
     k per (SA2, type) = target / capacity; pairs with no base-year stock take the state-and-type median (national-by-type
     where the state has none); in pairs outside the national band (one-third to three times the national median), cells that
     carry the type keep the SA2's k and every other cell takes the state-and-type median.
  6. With --hir-cells, step 5 runs with base-year HIR in place: on Beef and Sheep natural land cells inside existing HIR
     projects (registered up to the baseline FY; LUTO's base-year adoption at RESFACTOR 1), capacity is scaled by
     1 - (1 - HIR stocking factor) x the adopted share, so the type's SA2 head still equals its target and the head the
     HIR area no longer carries moves to the type's other cells. Typing (step 4) uses the full capacity. The recalibrated
     k applies only to the cells that carry the type in the base year in those (SA2, type) pairs; the band, the
     state-and-type medians and every other cell's k are step 5's, before the adjustment. The k file then matches runs
     with HIR_BASE_YEAR_PROJECTS set and HIR_PRODUCTIVITY_CONTRIBUTION equal to the factor.
Outputs in --out: lumap.h5, lmmap.h5 (layout of the 2010-base files), stubble_dse_ha.h5 (Series STUBBLE_DSE_HA, float32),
lvstk_k.h5 (BEEF, SHEEP, DAIRY float32), and a NetCDF copy of each of the four in the jinzhu layout (nc_layout.py:
lumap.nc, lmmap.nc, stubble_dse_ha.nc, lvstk_k.nc with variables BEEF, SHEEP, DAIRY); livestock_dse_ratings.csv, livestock_recon_state_type.csv (NIR head = allocated
pasture and stubble head + named rows; NIR and estimated feedlot head), livestock_recon_sa2.csv (with the town-SA2 pooling
flag, target and k before pooling, and the estimated feedlot head), livestock_hir_recalibration.csv (per SA2 and type with HIR
area: HIR capacity share, k without and with HIR, head moved onto other cells, band flags); provenance.csv
(last). No feed_req.h5.

Run as a plain script from the repository root (no luto imports):

    python input/Nick_build/scripts/livestock_map.py --input input/Nick_build/inherited_2010 --nlum-map input/Nick_build/nlum_map
        --pasture input/Nick_build/pasture_productivity/pasture_kg_dm_ha.h5 --hir-cells input/Nick_build/ag_mgt_caps --out input/Nick_build/livestock_map
        --cache input/Nick_build/_cache [--offline] [--live-check] [--source NAME=PATH]

Dependencies: numpy, pandas, PyTables, openpyxl, rasterio, geopandas, scipy, xarray, netCDF4. No luto imports
(nc_layout.py is a sibling module).
Memory: peak resident memory about 2.0 GB on the full grid (measured October 2026); run on a machine with at least 8
GB of RAM.
Output values are deterministic; HDF5 file bytes carry a group timestamp, so provenance records dataset content hashes.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import time
import urllib.request
from importlib import metadata

import numpy as np
import pandas as pd
import rasterio

import nc_layout as NC

# ----------------------------------------------------------------------------- sources and pins
ANGA = 'https://greenhouseaccounts.climatechange.gov.au/api/StatesOutputTimeSeries?sectorid={s}&urlencodedlocationlist=3&fuelid=1&gasid=34'
ANGA_NODES = {684: 'dairy cattle enteric', 5811: 'beef cattle - pasture enteric', 706: 'sheep: rams', 707: 'sheep: wethers',
              708: 'sheep: maiden ewes', 709: 'sheep: breeding ewes', 710: 'sheep: other ewes', 711: 'sheep: lambs and hoggets'}
ANGA_PINS = {684: '030809d4da6532e38c328d08e7c888f792230f0bca1bd5e630945fab179b2254',
             5811: 'c86289c8bd6dea6661c8f578efc09c910499b8f8414135157fdaafd00f038288',
             706: 'b3b4bebafa594e4706f700f83497b890f60d48f7c96f4858493656781975acab',
             707: '56aa85099292aa1cdccecc60ad3beaf8df2fa8d51c920e950419b427a9b17701',
             708: '4aed8a91bb369fed2a85cbbdf59f9a6c3bfd8d79282c74a817d9fd9537d619da',
             709: 'bf7ef9052b99e65a15bb6be09f91e52bcb94eb216c7366857110c2a32e31d938',
             710: '32537fec14bb6f57c7dfd12f7bd21995437b65f63e7b6d323f66f907843c163b',
             711: '3f78c921c15a3e658d9bb0c55792fe344cd4dd9f7607fa1c8cc495de44c78d35'}
SOURCES = {   # name: (url, sha256 of the raw file or response, note)
    'abs_7121_2020_21': ('https://www.abs.gov.au/statistics/industry/agriculture/agricultural-commodities-australia/2020-21/'
                         'AGCDCASGS202021.xlsx', '0fdbd3e398b598a24c8bc315e5bd7cae36e8ea4fb6f6cea5a7b745e0950bedf5',
                         'ABS 7121.0 Agricultural Commodities, Australia, 2020-21, ASGS Ed 3 cube, CC BY 4.0'),
    'nir_livestock': ('https://www.dcceew.gov.au/sites/default/files/documents/'
                      'activity-table-1990-2024-agriculture-livestock-national.xlsx',
                      '669ac799188e97f993dc85662886856295fe2dc0e68e26590058d765e97efa59',
                      'DCCEEW NIR 2024 activity table, agriculture livestock, CC BY 4.0'),
    'sa2_2021': ('https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/'
                 'jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA94.zip',
                 '87345a241c20d64068a012616adb74c0eada6cc47ce2a897d62d78026deaf0cc',
                 'ABS ASGS Edition 3 SA2 2021 GDA94, CC BY 4.0'),
    'alum_2020_21': ('https://www.agriculture.gov.au/sites/default/files/documents/NLUM_v7_1_250m_ALUMV8_2020_21_alb_20260814.zip',
                     '6e937a1cf1e99bffc251b684cdd3f55429830fa0c7790b7191abfef995851fb7',
                     'ABARES NLUM v7.1 250 m ALUM v8 2020-21 (release 2026-08-14; the file nlum_map.py pins), '
                     'CC BY 4.0: feedlot area (ALUM 5.2.2) by SA2'),
}
for _n, _p in ANGA_PINS.items():
    SOURCES[f'anga_L3_S{_n}_G34'] = (ANGA.format(s=_n), _p, f'National Greenhouse Accounts API (NIR 2024), Australia, '
                                     f'{ANGA_NODES[_n]}, CH4 (Gg); raw response pinned (changing source)')
INHERITED = ['real_area.h5', 'safe_pur_natl.h5', 'safe_pur_modl.h5', 'state_id.npy', 'NLUM_2010-11_mask.tif', 'lumap.h5', 'lmmap.h5']
NCELL = 6956407
FY = '2020-21'

# ============================================================================= reference table (hand-entered)
# Each value with its source. Constants of the method rather than data; changing any is a ruling.
REF = {
    # Charmley et al. (2015), as NIR 2024 Vol 1 eq. 3.A.1a_3 / 3.A.1b_3: CH4 (g/head/day) = 20.7 x DMI (kg/day)
    'charmley_g_ch4_per_kg_dmi': 20.7,
    # Howden et al. (1994), as NIR 2024 Vol 1 sheep enteric equation: CH4 (kg/head/day) = 0.0188 x DMI + 0.00158
    'howden_slope': 0.0188, 'howden_intercept': 0.00158,
    # LUTO grassfed factors by type, unchanged from the 2010 build (luto settings)
    'grassfed': {'BEEF': .85, 'SHEEP': .85, 'DAIRY': .65},
    # Stubble carrying rate, DSE-year per ha, all states: MLA "Grazing modern stubbles" (2020), about 150 sheep
    # grazing days per ha on fresh wheat stubble (mature ewes taken as 1 DSE) = 2.5 DSE/ha x about 60 days; 150/365
    'stubble_dse_year_ha': 0.410959,
    # Six cells of SA2 511041285 (Geraldton, WA) carry stubble rate 0 (no LUTO sheep economics, 2010 build)
    'stub_zero_cells': [4600950, 4604849, 4608748, 4608750, 4612649, 4624347], 'geraldton_sa2': 511041285,
    'geraldton_reason': 'no cell to carry it: no LUTO sheep economics on its cereal cells (2010 build)',
    # Stocking on the adopted share of a cell under HIR, relative to full stocking: LUTO settings
    # HIR_PRODUCTIVITY_CONTRIBUTION (0.5). k is calibrated with base-year HIR in place, so the model's setting must match
    'hir_stocking_factor': 0.5,
    # Feedlot cattle (estimated): the NIR state totals for beef are "beef cattle - pasture", while ABS SA2 meat cattle
    # include feedlot cattle. Before scaling, each SA2's ABS meat cattle lose an estimate of its feedlot head: the NIR state
    # feedlot head (the "Annual equivalent numbers" rows of the "Number of feedlot cattle" block: domestic, mid-fed and
    # long-fed) x the SA2's share of the state's NLUM 2020-21 feedlot area (ALUM 5.2.2); used to calibrate k only (the
    # typing uses the census-scaled targets), and limited as in allocate().
    'nir_feedlot_block': ('Number of feedlot cattle', 'Number of sheep'), 'nir_feedlot_rows': 'Annual equivalent numbers',
    'alum_feedlot': '5.2.2',
    # Town SA2s: an (SA2, type) pair above the band in an SA2 smaller than this area (ha) is calibrated with its SA3.
    # Basis: ABS designs SA2s for 3,000 to 25,000 residents, so a small SA2 is a settlement, and ABS counts a holding's
    # stock at its location; on the 2021 data every above-band pair beyond 4.5 times the band edge lies in an SA2 under
    # 20,000 ha, and the pairs above the band in larger SA2s are capacity gaps, left as calibrated.
    'town_sa2_max_ha': 20000.,
}
# ============================================================================= end of reference table

TYPES = ('BEEF', 'SHEEP', 'DAIRY')
T = ['beef', 'sheep', 'dairy']
ST_NAME = {1: 'NSW', 2: 'Vic', 3: 'Qld', 4: 'SA', 5: 'WA', 6: 'Tas', 7: 'NT', 8: 'ACT', 9: 'OT'}
LU_MOD = [1, 14, 5]               # beef, sheep, dairy on modified land (ag_landuses.csv row)
LU_NAT = [2, 15, 6]               # beef, sheep, dairy on natural land
LV = LU_MOD + LU_NAT
U_MOD, U_NAT = 22, 23
CEREAL_LU = [18, 25]
TYPE_CAT = {12: 70, 13: 71, 22: 72, 23: 73}
CAT_R6, CAT_RETYPE, CAT_UNALLOC = 75, 76, 77


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
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (luto-livestock-map/1.0)',
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


# ----------------------------------------------------------------------------- step 1: feed ratings
def nir_populations(xlsx):
    """NIR activity table, Australia sheet, FY2020-21: population by group and class."""
    import openpyxl
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    R = list(wb['Australia'].iter_rows(values_only=True))
    hdr = R[5]
    ycols = {i: h for i, h in enumerate(hdr) if isinstance(h, str) and '-' in h and h[:2] in ('19', '20')}
    rows, grp, sub = [], None, None
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
            if fy == FY:
                rows.append(dict(group=grp, cls=cls.strip(), value=r[i]))
    return pd.DataFrame(rows)


def anga_value(path):
    d = json.load(open(path, encoding='utf-8'))
    assert d['error'] == '', d['error']
    v = [x['v'] for x in d['data'] if x['date'] == int(FY[:4]) + 1]
    assert len(v) == 1, (path, v)
    return float(v[0])


def feed_ratings(zips):
    """National FY2020-21 intake (kg DM/head/day), the DSE rating per head (1 kg DM per DSE per day)."""
    P = nir_populations(zips['nir_livestock'])
    ch4 = {n: anga_value(zips[f'anga_L3_S{n}_G34']) for n in ANGA_NODES}
    rows = []
    for an, (g, node) in {'DAIRY': ('Number of dairy cattle', 684), 'BEEF': ('Number of beef cattle - pasture', 5811)}.items():
        pp = P[P.group == g].value.sum()
        M = ch4[node] * 1e6 / pp / 365
        rows.append(dict(animal=an, pop=pp, kg_ch4_head_day=M, intake_kg_dm_day=M * 1000 / REF['charmley_g_ch4_per_kg_dmi'],
                         basis='Charmley: NIR enteric CH4 / population / 365'))
    CL = {706: 'Rams', 707: 'Wethers', 708: 'Maiden Ewes (intended for breeding)', 709: 'Breeding Ewes', 710: 'Other Ewes',
          711: 'Lambs and Hoggets'}
    dm = []
    for node, cls in CL.items():
        n = P[(P.group == 'Number of sheep') & (P.cls == cls)].value.sum()
        M = ch4[node] * 1e6 / (n * 365)
        dm.append(dict(cls=cls, pop=n, M=M, I=(M - REF['howden_intercept']) / REF['howden_slope']))
    D = pd.DataFrame(dm)
    rows.append(dict(animal='SHEEP', pop=D['pop'].sum(), kg_ch4_head_day=np.nan, intake_kg_dm_day=(D.I * D['pop']).sum() / D['pop'].sum(),
                     basis='Howden by class, flock-weighted'))
    R = pd.DataFrame(rows)
    return {t: float(R.set_index('animal').intake_kg_dm_day[t]) for t in TYPES}, R


# ----------------------------------------------------------------------------- step 2: head targets
def nir_state_heads(xlsx):
    """NIR state populations FY2020-21 by type (states, not Australia), summed over classes."""
    out = []
    for st in ['NSW', 'Qld', 'Vic', 'WA', 'SA', 'NT', 'Tas', 'ACT']:
        d = pd.read_excel(xlsx, sheet_name=st, header=None)
        h = d.iloc[5]
        ycols = [c for c in d.columns if isinstance(h[c], str) and '-' in h[c] and h[c][:2] in ('19', '20') and h[c] == FY]
        for grp, a, b in [('beef', 'Number of beef cattle - pasture', 'Number of dairy cattle'),
                          ('dairy', 'Number of dairy cattle', 'Number of feedlot'), ('sheep', 'Number of sheep', 'Number of pigs')]:
            i0 = d.index[d[0].astype(str).str.startswith(a)][0]
            i1 = d.index[d[0].astype(str).str.startswith(b)][0]
            for i in range(i0 + 1, i1):
                for c in ycols:
                    out.append(dict(region=st, group=grp, value=pd.to_numeric(d.iloc[i, c], errors='coerce')))
    g = pd.DataFrame(out).groupby(['region', 'group']).value.sum().unstack()
    inv = {v: k for k, v in ST_NAME.items()}
    g.index = [inv[r] for r in g.index]
    return g[T].sort_index().astype(float)


def nir_state_feedlot(xlsx):
    """NIR state feedlot cattle FY2020-21: the "Annual equivalent numbers" rows (domestic, mid-fed, long-fed) of the
    "Number of feedlot cattle" block, by state."""
    a, b = REF['nir_feedlot_block']
    out = {}
    inv = {v: k for k, v in ST_NAME.items()}
    for st in ['NSW', 'Qld', 'Vic', 'WA', 'SA', 'NT', 'Tas', 'ACT']:
        d = pd.read_excel(xlsx, sheet_name=st, header=None)
        h = d.iloc[5]
        yc = [c for c in d.columns if h[c] == FY]
        assert len(yc) == 1, (st, yc)
        i0 = d.index[d[0].astype(str).str.startswith(a)][0]
        i1 = d.index[d[0].astype(str).str.startswith(b)][0]
        rows = [i for i in range(i0 + 1, i1) if str(d.iloc[i, 3]).strip() == REF['nir_feedlot_rows']]
        assert len(rows) == 3, (st, rows)
        out[inv[st]] = float(pd.to_numeric(d.iloc[rows, yc[0]], errors='coerce').fillna(0).sum())
    return pd.Series(out).sort_index()


def feedlot_area_sa2(zp, mask_path, sa2):
    """NLUM 2020-21 feedlot area (ALUM 5.2.2) by SA2 code, ha: each 250 m pixel of the class is placed in the LUTO cell
    containing its centre (reprojected to the LUTO grid) and takes that cell's SA2."""
    import io as _io
    import zipfile
    from pyproj import Transformer
    z = zipfile.ZipFile(zp)
    base = 'NLUM_v7_1_250m_ALUMV8_2020_21_alb'
    vat = pd.read_csv(_io.BytesIO(z.read(base + '.csv')))
    vals = vat.Value[vat.TERTV8.astype(str).str.split(' ').str[0] == REF['alum_feedlot']].tolist()
    assert vals, 'no feedlot class in the NLUM table'
    with rasterio.open(mask_path) as r:
        mk = r.read(1) == 1
        mtr, mcrs = r.transform, r.crs
    with rasterio.open('zip://' + zp.replace('\\', '/') + '!' + base + '.tif') as r:
        a = r.read(1)
        ntr, ncrs = r.transform, r.crs
    px_ha = abs(ntr.a * ntr.e) / 1e4
    rw, cl = np.nonzero(np.isin(a, vals))
    del a
    x, y = ntr * (cl + .5, rw + .5)
    lon, lat = Transformer.from_crs(ncrs, mcrs, always_xy=True).transform(np.asarray(x), np.asarray(y))
    mc, mr = ~mtr * (lon, lat)
    mr, mc = np.floor(mr).astype(int), np.floor(mc).astype(int)
    ok = (mr >= 0) & (mr < mk.shape[0]) & (mc >= 0) & (mc < mk.shape[1])
    cidx = np.full(mk.shape, -1, np.int64)
    cidx[mk] = np.arange(int(mk.sum()))
    ci = np.full(len(mr), -1)
    ci[ok] = cidx[mr[ok], mc[ok]]
    return pd.Series(px_ha, index=sa2[ci[ci >= 0]]).groupby(level=0).sum(), dict(pixels=len(rw), on_cells=int((ci >= 0).sum()),
                                                                                 px_ha=px_ha)


def feedlot_head_sa2(area, nir_fl):
    """Estimated feedlot head by SA2 = NIR state feedlot head x the SA2's share of the state's NLUM feedlot area."""
    st = pd.Series(area.index // 10**8, index=area.index)
    tot = area.groupby(st).sum()
    est = area * st.map(nir_fl).fillna(0.) / st.map(tot)
    return est[est > 0]


def abs_livestock(xlsx):
    """ABS 7121.0 2020-21 'Table 1': SA2 head by type with high-RSE flags and names; state head by type (s_abs_extract)."""
    import openpyxl
    ws = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)['Table 1']
    rows = list(ws.iter_rows(values_only=True))
    hi = next(i for i, r in enumerate(rows) if r[0] == 'Region code')
    df = pd.DataFrame(rows[hi + 1:], columns=['code', 'name', 'commodity_code', 'commodity', 'value', 'flag', 'nb', 'nb_rse'])

    def codelen(x):
        try:
            return len(str(int(x)))
        except (TypeError, ValueError):
            return -1
    df['codelen'] = df['code'].apply(codelen)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    full = df[df.codelen == 9][['code', 'name']].drop_duplicates()
    LAB = {'Livestock - Meat cattle - Total (no.)': 'beef', 'Livestock - Dairy cattle - Total (no.)': 'dairy',
           'Livestock - Sheep and lambs - Total (no.)': 'sheep'}
    d = df[(df.codelen == 9) & df.commodity.isin(list(LAB))].copy()
    d['metric'] = d.commodity.map(LAB)
    pv = d.pivot_table(index=['code', 'name'], columns='metric', values='value', aggfunc='first')
    pf = d.pivot_table(index=['code', 'name'], columns='metric', values='flag', aggfunc='first')
    pf.columns = [f'{c}_flag' for c in pf.columns]
    res = full.merge(pv.join(pf).reset_index(), on=['code', 'name'], how='left')
    for m in T:
        for c in (m, f'{m}_flag'):
            if c not in res.columns:
                res[c] = np.nan
    res['code'] = res['code'].astype(np.int64)
    ab = res.set_index('code')
    H = ab[T].fillna(0.0).astype(float)
    RSE = ab[[t + '_flag' for t in T]].isin(['*', '**'])
    RSE.columns = T
    s = df[(df.codelen == 1) & df.commodity.isin(list(LAB))].copy()
    s['region_code'] = s['code'].astype(int)
    out = {t: s[s.commodity == lab].set_index('region_code').value for lab, t in LAB.items()}
    st = pd.DataFrame(out)[T].fillna(0.0)
    return H, RSE, ab['name'], st.drop(index=0)


def targets(H_abs, abs_st, nir_st, FL=None, nir_fl=None, floor=None):
    """SA2 target = ABS SA2 head x (NIR state total / ABS state total) per type; per-state table. With FL (estimated
    feedlot head by SA2), beef is scaled from pasture head: ABS SA2 meat cattle minus FL, not below `floor` (by SA2; 0
    where not given), against the ABS state total minus the head subtracted in the state."""
    st = H_abs.index // 10**8
    fl = (FL.reindex(H_abs.index).fillna(0.) if FL is not None else pd.Series(0., index=H_abs.index))
    fo = floor.reindex(H_abs.index).fillna(0.) if floor is not None else pd.Series(0., index=H_abs.index)
    sub = np.minimum(fl, np.maximum(H_abs['beef'] - fo, 0.))
    H_abs = H_abs.copy()
    H_abs['beef'] = H_abs['beef'] - sub
    sub_st = sub.groupby(st).sum()
    abs_st = abs_st.copy()
    abs_st['beef'] = abs_st['beef'] - sub_st.reindex(abs_st.index).fillna(0.)
    ratio = pd.DataFrame(index=nir_st.index, columns=T, dtype=float)
    for t in T:
        for s in nir_st.index:
            a = abs_st.loc[s, t] if s in abs_st.index else 0.
            ratio.loc[s, t] = nir_st.loc[s, t] / a if a > 0 else 0.
    tgt = H_abs.copy()
    for t in T:
        tgt[t] = H_abs[t].to_numpy() * np.array([ratio.loc[s, t] if s in ratio.index else 0. for s in st])
    rows = []
    for s in nir_st.index:
        for t in T:
            m = st == s
            rows.append(dict(state=s, state_name=ST_NAME[s], type=t, state_total=nir_st.loc[s, t],
                             abs_state=abs_st.loc[s, t] if s in abs_st.index else 0., ratio=ratio.loc[s, t],
                             abs_sa2_sum=H_abs.loc[m, t].sum(), target_sum=tgt.loc[m, t].sum(),
                             nir_feedlot_head=(nir_fl.get(s, 0.) if nir_fl is not None and t == 'beef' else 0.),
                             feedlot_head_estimated=(FL[FL.index // 10**8 == s].sum() if FL is not None and t == 'beef' else 0.),
                             feedlot_head_subtracted=(sub[m].sum() if t == 'beef' else 0.)))
    S = pd.DataFrame(rows)
    S['row_state_total_minus_sa2_sum'] = S.state_total - S.target_sum
    return tgt, S, pd.DataFrame({'feedlot_head_estimated': fl, 'feedlot_head_subtracted': sub})


def wmedian(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


def lu2t(lu):
    t = np.full(lu.shape, -1, np.int8)
    for i in range(3):
        t[(lu == LU_MOD[i]) | (lu == LU_NAT[i])] = i
    return t


def fix_state_ties(sa2, ST, mask_path):
    """A cell whose SA2 state digit disagrees with state_id takes the SA2 of the nearest cell (grid distance)
    whose SA2 digit equals that state_id and agrees with its own state_id."""
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


def gcc_flags(shp_zip):
    import geopandas as gpd
    g = gpd.read_file('zip://' + shp_zip.replace('\\', '/') + '!SA2_2021_AUST_GDA94.shp', ignore_geometry=True)
    g = g[g['SA2_CODE21'].str.fullmatch(r'[0-9]{9}')]
    return pd.Series(g['GCC_CODE21'].str[1].eq('G').to_numpy(), index=g['SA2_CODE21'].astype(np.int64).to_numpy())


# ----------------------------------------------------------------------------- steps 3 to 5: typing and k
def hir_project_fraction(hir_dir, n):
    """Share of each cell inside the existing HIR projects registered up to the baseline FY, clipped at the whole cell:
    the cells and rule of LUTO's base-year HIR adoption (Data.load_hir_base_year_adoption) at RESFACTOR 1."""
    cells = pd.read_csv(os.path.join(hir_dir, 'hir_project_cells.csv'))
    baseline_fy = int(pd.read_csv(os.path.join(hir_dir, 'hir_scaling.csv')).iloc[0]['baseline_fy'])
    frac = cells.loc[cells['fy_registered'] <= baseline_fy].groupby('cell')['fraction'].sum().clip(upper=1)
    out = np.zeros(n)
    out[frac.index.to_numpy()] = frac.to_numpy()
    return out, baseline_fy


def allocate(DSEd, A, PK, SN, SM, ST, lu0, lm0, cat0, sa2, H_abs, RSE, NAMES, abs_st, nir_st, GCC, HIRF=None, FL=None,
             nir_fl=None):
    N = len(A)
    GF = np.array([REF['grassfed'][t] for t in TYPES])
    DSE = np.array([DSEd[t] for t in TYPES])
    STATE = sa2 // 10**8
    NAT = np.isin(lu0, [2, 6, 15, 23])
    IRR = lm0 == 1
    cap = np.nan_to_num(PK * np.where(NAT, SN, SM) * A * np.where(IRR, 2., 1.) / 365.)
    t0 = lu2t(lu0)
    CAT0LV = (cat0 == 0) & (t0 >= 0)
    PERS = (np.isin(cat0, [10, 11, 20, 21, 51, 0])) & (t0 >= 0)
    ASG = np.isin(cat0, [12, 13, 22, 23])
    KEEPU = (cat0 == 10) & (lu0 == U_NAT)
    OUTLV = (t0 >= 0) & ~PERS
    assert not (ASG & (t0 >= 0)).any() and not (KEEPU & PERS).any() and not OUTLV.any()
    CER = np.isin(lu0, CEREAL_LU)
    TGT, STATE_TAB_TYP, _ = targets(H_abs, abs_st, nir_st)          # census-scaled targets: the typing uses these
    CODES = np.unique(np.r_[sa2, TGT.index.to_numpy()])
    sidx = np.searchsorted(CODES, sa2)
    NS = len(CODES)
    TG0 = np.zeros((NS, 3))
    TG0[np.searchsorted(CODES, TGT.index.to_numpy())] = TGT[T].to_numpy()
    RS = np.zeros((NS, 3), bool)
    RS[np.searchsorted(CODES, RSE.index.to_numpy())] = RSE[T].to_numpy()
    ABSV = np.zeros((NS, 3))
    ABSV[np.searchsorted(CODES, H_abs.index.to_numpy())] = H_abs[T].to_numpy()
    CODE_STATE = CODES // 10**8
    # stubble carrying rate on every cell of an SA2 with a sheep target; 0 on the six Geraldton cells
    SHEEP_SA2 = TG0[:, 1] > 0
    STUB_DSE_HA = np.where(SHEEP_SA2[sidx], pd.Series(STATE).map(lambda s: REF['stubble_dse_year_ha']).fillna(0.).to_numpy(), 0.)
    STUB_DSE_HA_PRE = STUB_DSE_HA.copy()
    Z = REF['stub_zero_cells']
    assert (sa2[Z] == REF['geraldton_sa2']).all() and (STUB_DSE_HA_PRE[Z] > 0).all()
    STUB_DSE_HA[Z] = 0.
    stubcap = np.where(CER, STUB_DSE_HA * A, 0.)
    STUBCAP_PRE_S = np.bincount(sidx, weights=np.where(CER, STUB_DSE_HA_PRE * A, 0.), minlength=NS)
    STUBCAP_S = np.bincount(sidx, weights=stubcap, minlength=NS)

    def greedy(idx, fixed, Ts, stub_s):
        D = Ts * DSE * GF
        E = D > 0
        ty = fixed.copy()
        asg = np.where(fixed < 0)[0]
        if len(asg) == 0 or not E.any():
            return ty
        w = D / D.sum()
        c = cap[idx]
        mod = ~NAT[idx]
        irr = IRR[idx]
        live = np.zeros(len(idx), bool)
        for i in range(3):
            live |= (fixed == i) & E[i]
        sc = stub_s if E[1] else 0.
        Ctot = c[live].sum() + c[asg].sum() + sc
        C = np.array([c[fixed == i].sum() if E[i] else 0. for i in range(3)])
        C[1] += sc
        taken = np.zeros(len(idx), bool)
        if E[2]:
            cand = asg[mod[asg]]
            cand = cand[np.lexsort((-c[cand], -irr[cand].astype(int)))]
            for j in cand:
                if C[2] >= w[2] * Ctot:
                    break
                ty[j] = 2
                C[2] += c[j]
                taken[j] = True
        pool = [i for i in (0, 1) if E[i]]
        rest = asg[~taken[asg]]
        rest = rest[np.argsort(-c[rest], kind='stable')]
        for j in rest:
            if not pool:
                continue
            i = max(pool, key=lambda i: w[i] * Ctot - C[i])
            ty[j] = i
            C[i] += c[j]
        return ty

    def calib(ty, TG, capx=cap):
        ch = np.zeros((NS, 3))
        nc = np.zeros((NS, 3), np.int64)
        for i in range(3):
            m = ty == i
            ch[:, i] = np.bincount(sidx[m], weights=capx[m] / (DSE[i] * GF[i]), minlength=NS)
            nc[:, i] = np.bincount(sidx[m], minlength=NS)
        ch[:, 1] += STUBCAP_S / (DSE[1] * GF[1])
        with np.errstate(divide='ignore', invalid='ignore'):
            k = np.where(ch > 0, TG / ch, np.nan)
        return k, ch, nc

    def nat_medians(k, TG):
        return np.array([wmedian(k[np.isfinite(k[:, i]) & (TG[:, i] > 0), i], TG[np.isfinite(k[:, i]) & (TG[:, i] > 0), i])
                         for i in range(3)])

    def retype_mask(TG):
        return PERS & (TG[sidx, np.maximum(t0, 0)] == 0)

    def typing(TG):
        RT = retype_mask(TG)
        ty = np.full(N, -1, np.int8)
        ty[PERS & ~RT] = t0[PERS & ~RT]
        cells = np.where(PERS | ASG)[0]
        cells = cells[np.argsort(sidx[cells], kind='stable')]
        u, st_, cn = np.unique(sidx[cells], return_index=True, return_counts=True)
        for s, a, n in zip(u, st_, cn):
            idx = cells[a:a + n]
            ty[idx] = greedy(idx, ty[idx].copy(), TG[s], STUBCAP_S[s])
        return ty

    def ruling6(ty, TG):
        k1, ch1, _ = calib(ty, TG)
        med1 = nat_medians(k1, TG)
        UP = np.broadcast_to(3 * med1, (NS, 3))
        QUAL = ((TG > 0) & np.isfinite(k1) & (k1 > UP)) | ((TG > 0) & ~(ch1 > 0))
        QUAL[:, 2] = False
        kc = np.where(KEEPU)[0]
        kc = kc[np.lexsort((-cap[kc], sidx[kc]))]
        ku, kst, kcn = np.unique(sidx[kc], return_index=True, return_counts=True)
        kpos = {s: (a, n) for s, a, n in zip(ku, kst, kcn)}
        ty = ty.copy()
        R6 = np.zeros(N, bool)
        ch = ch1.copy()
        for s in np.where(QUAL.any(axis=1))[0]:
            act = [i for i in (0, 1) if QUAL[s, i]]
            if s not in kpos:
                continue
            a, n = kpos[s]
            for j in kc[a:a + n]:
                if not act:
                    break
                if cap[j] <= 0:
                    continue
                kk = {i: (TG[s, i] / ch[s, i] if ch[s, i] > 0 else np.inf) / UP[s, i] for i in act}
                i = max(act, key=lambda i: kk[i])
                ty[j] = i
                R6[j] = True
                ch[s, i] += cap[j] / (DSE[i] * GF[i])
                if TG[s, i] / ch[s, i] <= UP[s, i]:
                    act.remove(i)
        return ty, R6, k1, UP

    def per_type_k(k, TG):
        mt = (TG > 0) & np.isfinite(k)
        K = np.where(mt, k, np.nan)
        nat = np.array([wmedian(k[mt[:, i], i], TG[mt[:, i], i]) for i in range(3)])
        stmed = {}
        for s in np.unique(CODE_STATE):
            for i in range(3):
                m = mt[:, i] & (CODE_STATE == s)
                stmed[(int(s), i)] = wmedian(k[m, i], TG[m, i]) if m.any() else np.nan
        for i in range(3):
            for s in np.unique(CODE_STATE):
                m = ~mt[:, i] & (CODE_STATE == s)
                v = stmed[(int(s), i)]
                K[m, i] = v if np.isfinite(v) else nat[i]
        assert np.isfinite(K).all()
        return K, nat, stmed

    TG = TG0
    ty = typing(TG)
    ty, R6, k1, UP = ruling6(ty, TG)
    GREEDY = ASG & (ty >= 0)
    RT = retype_mask(TG)
    RETYPED = RT & (ty >= 0)
    UNALLOC = RT & (ty < 0)
    lu1, cat1 = lu0.copy(), cat0.copy()
    for i in range(3):
        lu1[(GREEDY | RETYPED) & NAT & (ty == i)] = LU_NAT[i]
        lu1[(GREEDY | RETYPED) & ~NAT & (ty == i)] = LU_MOD[i]
        lu1[R6 & (ty == i)] = LU_NAT[i]
    for c, v in TYPE_CAT.items():
        cat1[GREEDY & (cat0 == c)] = v
    lu1[UNALLOC & NAT] = U_NAT
    lu1[UNALLOC & ~NAT] = U_MOD
    cat1[RETYPED] = CAT_RETYPE
    cat1[UNALLOC] = CAT_UNALLOC
    cat1[R6] = CAT_R6
    assert not ((GREEDY | RETYPED) & NAT & (ty == 2)).any()
    assert (NAT[RETYPED] == np.isin(lu1[RETYPED], LU_NAT)).all() and (NAT[UNALLOC] == (lu1[UNALLOC] == U_NAT)).all()
    assert (np.isin(lu1, CEREAL_LU) == CER).all(), 'typing must not change cereal cells'
    # ---- calibration targets. The typing above used the census-scaled targets. k is calibrated on beef from pasture
    # head: each SA2's ABS meat cattle lose the estimated feedlot head before scaling; in an SA2 with typed beef cells the
    # subtraction stops where k on those cells would fall below the lower edge of the band (one-third of the national
    # median, at the census-scaled ratio), so no typed beef cell is left without stock; where those cells have no
    # capacity nothing is subtracted
    TGtyp = TG
    kT, chT, ncT = calib(ty, TGtyp)
    loT = nat_medians(kT, TGtyp)[0] / 3
    rat0 = STATE_TAB_TYP[STATE_TAB_TYP.type == 'beef'].set_index('state').ratio.astype(float)
    r0 = pd.Series(CODE_STATE).map(rat0).fillna(0.).to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        fmin = np.where(ncT[:, 0] > 0, np.where((chT[:, 0] > 0) & (r0 > 0), loT * chT[:, 0] / r0, np.inf), 0.)
    TGTc, STATE_TAB, FLT = targets(H_abs, abs_st, nir_st, FL, nir_fl, pd.Series(fmin, index=CODES))
    TG = np.zeros((NS, 3))
    TG[np.searchsorted(CODES, TGTc.index.to_numpy())] = TGTc[T].to_numpy()
    assert np.allclose(TG[:, 1:], TGtyp[:, 1:])
    # ---- town SA2s calibrated with their SA3: an (SA2, type) pair above the band (k on the typed cells before HIR) in an
    # SA2 under REF['town_sa2_max_ha'] takes the k of its SA3's pool (the SA3's target over its capacity, over the SA2s of
    # the SA3 with target and capacity of the type); the head its cells cannot carry at that k moves to the SA3's other
    # SA2s of the type in proportion to their target, so each keeps its relative k and the SA3 total is unchanged
    k0, ch0, _ = calib(ty, TG)
    med0 = nat_medians(k0, TG)
    SA2HA = np.bincount(sidx, weights=A, minlength=NS)
    POOL = (TG > 0) & (ch0 > 0) & (k0 > 3 * med0) & (SA2HA < REF['town_sa2_max_ha'])[:, None]
    SA3 = CODES // 10**4
    TGp = TG.copy()
    for i in range(3):
        for g in np.unique(SA3[POOL[:, i]]):
            mem = (SA3 == g) & (TG[:, i] > 0) & (ch0[:, i] > 0)
            tw, oth = mem & POOL[:, i], mem & ~POOL[:, i]
            ku = TG[mem, i].sum() / ch0[mem, i].sum()
            TGp[tw, i] = ku * ch0[tw, i]
            if oth.any():
                TGp[oth, i] = TG[oth, i] * (TG[mem, i].sum() - TGp[tw, i].sum()) / TG[oth, i].sum()
            assert np.isclose(TGp[mem, i].sum(), TG[mem, i].sum())
    TGpre, TG = TG, TGp
    # ---- k with base-year HIR in place: the typing above uses the full capacity (types unchanged); k is calibrated on the
    # capacity left after the HIR stocking factor on each cell's adopted share (HIR projects on Beef and Sheep natural land),
    # so the type's head in the SA2 still equals its target and the head the HIR area no longer carries moves to the other cells
    FH = REF['hir_stocking_factor']
    XA = np.where(np.isin(lu1, LU_NAT[:2]), HIRF, 0.) if HIRF is not None else np.zeros(N)
    capH = cap * (1. - (1. - FH) * XA)
    # The national band, the state-and-type medians (cells with no base-year stock of a type, flagged pairs) and k on every
    # cell outside the recalibrated pairs come from k before the HIR adjustment (kW); the recalibrated k applies only to the
    # base-year cells of the type in the (SA2, type) pairs with HIR area (out_of_band_k)
    kW, chW, _ = calib(ty, TG)
    k, chF, ncF = calib(ty, TG, capH)
    medW = nat_medians(kW, TG)
    lo = np.broadcast_to(medW / 3, (NS, 3))
    hi = np.broadcast_to(medW * 3, (NS, 3))
    INS = ty >= 0
    head = np.zeros(N)
    head_stub = np.zeros(N)
    kk = k[sidx, np.maximum(ty, 0)]
    m = INS & np.isfinite(kk)
    head[m] = kk[m] * capH[m] / (DSE[ty[m]] * GF[ty[m]])
    ks = k[sidx, 1]
    ms = CER & np.isfinite(ks) & (stubcap > 0)
    head_stub[ms] = ks[ms] * stubcap[ms] / (DSE[1] * GF[1])
    # per (SA2, type) with HIR area: capacity share on the HIR area, k without and with HIR, head the HIR area carries
    # without and with HIR, and the difference, which moves onto the type's other cells (and the stubble, for sheep)
    HR = []
    HP = np.zeros((NS, 3), bool)                      # the recalibrated (SA2, type) pairs
    for i in range(2):
        mi = INS & (ty == i)
        c_hir = np.bincount(sidx[mi], weights=cap[mi] * XA[mi] / (DSE[i] * GF[i]), minlength=NS)
        for s in np.where((c_hir > 0) & (TG[:, i] > 0) & (chF[:, i] > 0))[0]:
            HP[s, i] = True
            fw = 'above' if kW[s, i] > 3 * medW[i] else 'below' if kW[s, i] < medW[i] / 3 else ''
            fn = 'above' if k[s, i] > hi[s, i] else 'below' if k[s, i] < lo[s, i] else ''
            h0, h1 = kW[s, i] * c_hir[s], FH * k[s, i] * c_hir[s]
            HR.append(dict(sa2_code=CODES[s], sa2_name=NAMES.get(CODES[s], ''), state=CODE_STATE[s], type=T[i], target=TG[s, i],
                           hir_capacity_share=c_hir[s] / chW[s, i], k_without_hir=kW[s, i], k=k[s, i],
                           k_ratio=k[s, i] / kW[s, i], head_hir_area_without_hir=h0, head_hir_area=h1,
                           head_moved_onto_other_cells=h0 - h1, flag_without_hir=fw, flag=fn,
                           above_band_only_with_hir=(fn == 'above') and (fw != 'above')))
    HR = pd.DataFrame(HR, columns=['sa2_code', 'sa2_name', 'state', 'type', 'target', 'hir_capacity_share', 'k_without_hir', 'k',
                                   'k_ratio', 'head_hir_area_without_hir', 'head_hir_area', 'head_moved_onto_other_cells',
                                   'flag_without_hir', 'flag', 'above_band_only_with_hir'])
    # ---- SA2 rows: named rows and the national band flag
    ag = lu1 >= 0
    crop = ag & ~np.isin(lu1, LV + [U_MOD, U_NAT])
    unst = (lu1 == -1) | np.isin(lu1, [U_MOD, U_NAT])
    bc = lambda msk: np.bincount(sidx[msk], weights=A[msk], minlength=NS)
    sa2_area, ag_area, crop_area, unst_area = bc(np.ones(N, bool)), bc(ag), bc(crop), bc(unst)
    with np.errstate(divide='ignore', invalid='ignore'):
        crop_share = np.where(ag_area > 0, crop_area / ag_area, 0.)
        gap_share = np.where(sa2_area > 0, unst_area / sa2_area, 0.)
    scope_any = (np.bincount(sidx[INS], minlength=NS) > 0) | (STUBCAP_S > 0)
    ng = np.stack([np.bincount(sidx[(GREEDY | RETYPED) & (ty == i)], minlength=NS) for i in range(3)], 1)
    nr6 = np.stack([np.bincount(sidx[R6 & (ty == i)], minlength=NS) for i in range(3)], 1)
    R = []
    for s in range(NS):
        for i, t in enumerate(T):
            tg = TG[s, i]
            n = ncF[s, i]
            if tg > 0:
                if chF[s, i] > 0:
                    rc = 'matched'
                elif i == 1 and STUBCAP_PRE_S[s] > 0 and STUBCAP_S[s] == 0 and n == 0:
                    rc = REF['geraldton_reason']
                elif not scope_any[s]:
                    rc = 'no cell to carry it: SA2 has no grazing cell in scope'
                elif n == 0:
                    rc = 'no cell to carry it: no cell of this type in SA2'
                else:
                    rc = 'no cell to carry it: cells of this type have zero capacity'
            else:
                rc = 'cells of type with SA2 target 0 (head 0)' if n > 0 else 'empty'
            if rc == 'empty' and ABSV[s, i] == 0 and TG0[s, i] == 0:
                continue
            R.append(dict(sa2_code=CODES[s], sa2_name=NAMES.get(CODES[s], ''), state=CODE_STATE[s], type=t, abs_head=ABSV[s, i],
                          target=tg, rse_ge25=RS[s, i], gcc=bool(GCC.get(CODES[s], False)), row_class=rc, cells=int(n),
                          cells_greedy=int(ng[s, i]), cells_r6=int(nr6[s, i]), k_first=k1[s, i], k=k[s, i], band_lo=lo[s, i],
                          band_hi=hi[s, i], crop_share=crop_share[s], gap_share=gap_share[s], sa2_ha=SA2HA[s],
                          pooled_with_sa3=bool(POOL[s, i]), target_census_scaled=TGtyp[s, i],
                          target_before_pooling=TGpre[s, i], k_before_pooling=k0[s, i],
                          feedlot_head_estimated=FLT.feedlot_head_estimated.get(CODES[s], 0.) if i == 0 else 0.,
                          feedlot_head_subtracted=FLT.feedlot_head_subtracted.get(CODES[s], 0.) if i == 0 else 0.))
    R = pd.DataFrame(R)
    key = sidx * 3 + np.maximum(ty, 0)
    hp = np.bincount(key[INS], weights=head[INS], minlength=NS * 3)
    hs = np.bincount(sidx, weights=head_stub, minlength=NS)
    TI = R.type.map({t: i for i, t in enumerate(T)}).to_numpy()
    rs = np.searchsorted(CODES, R.sa2_code.to_numpy())
    R['head_pasture'] = hp[rs * 3 + TI]
    R['head_stubble'] = np.where(TI == 1, hs[rs], 0.)
    R['head_alloc'] = R.head_pasture + R.head_stubble
    R['residual'] = R.target - R.head_alloc
    mt = R.row_class == 'matched'
    assert np.allclose(R.loc[mt, 'residual'], 0, atol=1e-6 * np.maximum(1, R.loc[mt, 'target']).max()), 'matched rows must close'
    assert (R.loc[~mt, 'head_alloc'] == 0).all()
    R['nat_median'] = medW[TI]
    R['flag'] = np.where(mt & (R.k > R.band_hi), 'above', np.where(mt & (R.k < R.band_lo), 'below', ''))
    R['rel'] = R.k / R.nat_median
    g = R[mt].groupby('sa2_code').rel
    wide_hi = (g.min() > 1) & (g.size() >= 2)
    wide_lo = (g.max() < 1) & (g.size() >= 2)
    c_sa2wide = np.where(R.flag == 'above', R.sa2_code.map(wide_hi).fillna(False).astype(bool),
                         R.sa2_code.map(wide_lo).fillna(False).astype(bool))

    def cause(r, wide):
        if r.flag == 'above':
            if r.crop_share > 0.5:
                return 'cropping SA2 (crop share > 0.5)'
            if r.rse_ge25 or r.gcc:
                return 'peri-urban or high RSE'
            if r.crop_share <= 0.5 and r.gap_share >= 0.2:
                return 'mapping gap (unstocked non-crop >= 20% of SA2 area)'
            return 'productivity layer against census (SA2-wide shortfall)' if wide else ''
        if r.flag == 'below':
            if r.rse_ge25 or r.gcc:
                return 'peri-urban or high RSE'
            if r.cells_greedy + r.cells_r6 == 0:
                return 'persistence surplus (all cells of type persistent)'
            return 'productivity layer against census (SA2-wide surplus)' if wide else ''
        return ''
    R['cause'] = [cause(r, w) for r, w in zip(R.itertuples(), c_sa2wide)]
    # ---- state reconciliation: NIR = allocated pasture head + stubble head + named rows
    named = ['no cell to carry it: SA2 has no grazing cell in scope', 'no cell to carry it: no cell of this type in SA2',
             'no cell to carry it: cells of this type have zero capacity', REF['geraldton_reason']]
    S = []
    for _, sr in STATE_TAB.iterrows():
        r = R[(R.state == sr.state) & (R.type == sr.type)]
        d = dict(state=sr.state, state_name=sr.state_name, type=sr.type, nir_state_total=sr.state_total, abs_state_total=sr.abs_state,
                 nir_over_abs=sr.ratio, nir_feedlot_head=sr.nir_feedlot_head,
                 feedlot_head_estimated=sr.feedlot_head_estimated, feedlot_head_subtracted=sr.feedlot_head_subtracted, **{'row: state total minus SA2 target sum (ABS SA2 rounding/suppression, scaled)':
                                           sr.row_state_total_minus_sa2_sum})
        for c in named:
            d['row: ' + c] = r.loc[r.row_class == c, 'target'].sum()
        d['allocated_pasture_head'] = r.head_pasture.sum()
        d['allocated_stubble_head'] = r.head_stubble.sum()
        h = HR[(HR.state == sr.state) & (HR.type == sr.type)]
        d['hir_head_moved_onto_other_cells'] = h.head_moved_onto_other_cells.sum()     # within allocated head; not a row
        d['residual'] = (d['nir_state_total'] - sum(v for kk_, v in d.items() if kk_.startswith('row: '))
                         - d['allocated_pasture_head'] - d['allocated_stubble_head'])
        S.append(d)
    S = pd.DataFrame(S)
    assert np.abs(S.residual).max() < 0.01, S.residual.abs().max()
    # ---- cell assertions
    pos = head > 0
    spos = head_stub > 0
    lvc = lu2t(lu1) >= 0
    tgt0 = INS & (TG[sidx, np.maximum(ty, 0)] == 0)
    assert not (lvc & tgt0).any() and (pos | tgt0 | (INS & (cap <= 0)))[lvc].all()
    assert (lu2t(lu1)[INS] == ty[INS]).all() and (INS == lvc).all() and not (head[~lvc] > 0).any() and not (spos & lvc).any()
    assert CER[spos].all() and not (spos & ~SHEEP_SA2[sidx]).any()
    K, Knat, Kst = per_type_k(kW, TG)                 # before the HIR adjustment; out_of_band_k sets the recalibrated pairs
    lm1 = np.where(np.isin(lu1, [U_MOD, U_NAT]), 0, lm0)          # LUTO has no irrigated unallocated land
    return dict(lu1=lu1, lm1=lm1, cat1=cat1, ty=ty, head=head, head_stub=head_stub, STUB_DSE_HA=STUB_DSE_HA, K=K, Knat=Knat,
                Kst=Kst, CODES=CODES, sidx=sidx, R=R, S=S, A=A, cap=cap, CER=CER, HR=HR, XA=XA, kH=k, HP=HP)


def out_of_band_k(o, sa2):
    """In matched (SA2, type) pairs flagged above or below the national band, cells that carry the type in the
    base-year allocation keep the SA2's k; every other cell of the SA2 takes the state-and-type median. K, the band and the
    medians are k before the HIR adjustment; in the pairs recalibrated for base-year HIR, the cells that carry the type in
    the base year (and the stubble cells, for sheep) take the recalibrated k, and no other cell changes."""
    K, CODES, sidx, ty, hstub, R = o['K'], o['CODES'], o['sidx'], o['ty'], o['head_stub'], o['R']
    N = len(ty)
    CST = CODES // 10**8

    def med(s, i):
        v = o['Kst'][(int(s), i)]
        return v if v is not None and np.isfinite(v) else o['Knat'][i]
    old = {c: K[sidx, i].astype(np.float32) for i, c in enumerate(TYPES)}
    new = {c: old[c].copy() for c in TYPES}
    carry = [ty == 0, (ty == 1) | (hstub > 0), ty == 2]
    F = R[(R.row_class == 'matched') & R.flag.isin(['above', 'below'])]
    cell_by_sa2 = pd.Series(np.arange(N)).groupby(sidx).apply(np.asarray)
    nmed = 0
    for _, r in F.iterrows():
        i = T.index(r.type)
        s = int(np.searchsorted(CODES, r.sa2_code))
        cells = cell_by_sa2.get(s, np.array([], int))
        tgt = cells[~carry[i][cells]]
        new[TYPES[i]][tgt] = np.float32(med(CST[s], i))
        nmed += len(tgt)
    # recalibrated (SA2, type) pairs (base-year HIR): the HIR k on the cells that carry the type in the base year only
    nhir = {}
    for s, i in zip(*np.nonzero(o['HP'])):
        cells = cell_by_sa2.get(s, np.array([], int))
        tgt = cells[carry[i][cells]]
        new[TYPES[i]][tgt] = np.float32(o['kH'][s, i])
        nhir[(int(CST[s]), TYPES[i])] = nhir.get((int(CST[s]), TYPES[i]), 0) + len(tgt)
    KA = pd.DataFrame({c: new[c].astype(np.float32) for c in TYPES}, index=pd.Index(np.arange(N, dtype=np.int64)))
    assert np.isfinite(KA.to_numpy()).all()
    return KA, dict(flagged_pairs=len(F), median_cells=nmed, hir_pairs=int(o['HP'].sum()), hir_cells=nhir)


# ----------------------------------------------------------------------------- provenance and outputs
def content_sha(a):
    a = np.ascontiguousarray(a)
    return sha(f'{a.dtype.str}|{a.shape}|'.encode() + a.tobytes())


def pkg_versions():
    out = {}
    for p in ('numpy', 'pandas', 'tables', 'openpyxl', 'rasterio', 'geopandas', 'pyogrio', 'scipy', 'xarray', 'netCDF4'):
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
    ap.add_argument('--nlum-map', required=True, help='output folder of nlum_map.py')
    ap.add_argument('--pasture', required=True, help='pasture_kg_dm_ha.h5 from pasture_productivity.py')
    ap.add_argument('--out', required=True, help='output folder (e.g. input/Nick_build/livestock_map)')
    ap.add_argument('--cache', required=True, help='shared raw-download cache (content-addressed by sha256)')
    ap.add_argument('--hir-cells', default=None, help='output folder of ag_mgt_caps.py (hir_project_cells.csv, '
                    'hir_scaling.csv): k is calibrated with base-year HIR in place, for runs with HIR_BASE_YEAR_PROJECTS set')
    ap.add_argument('--offline', action='store_true', help='fail instead of downloading on a cache miss')
    ap.add_argument('--live-check', action='store_true', help='also fetch every API response and stop if any differs from its pin')
    ap.add_argument('--source', action='append', default=[], metavar='NAME=PATH',
                    help='a local copy of a pinned source (must match its pin); NAME in ' + ', '.join(SOURCES))
    a = ap.parse_args()
    t0 = time.time()
    overrides = dict(s.split('=', 1) for s in a.source)
    assert set(overrides) <= set(SOURCES), set(overrides) - set(SOURCES)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    zips = {k: fetch(k, a.cache, a.offline, overrides) for k in SOURCES}
    if a.live_check:
        live_check([k for k in SOURCES if k.startswith('anga_')])
    inp = {f: os.path.join(a.input, f) for f in INHERITED}
    upstream = {f: os.path.join(a.nlum_map, f) for f in ('class_alloc_2020_21_lumap.npy', 'class_alloc_2020_21_lmmap.npy',
                                                          'class_alloc_2020_21_category.npy', 'sa2_2021_cells.npy')}
    upstream['pasture_kg_dm_ha.h5'] = a.pasture
    if a.hir_cells:
        for f in ('hir_project_cells.csv', 'hir_scaling.csv'):
            upstream[f] = os.path.join(a.hir_cells, f)
    in_sha = {f: sha_file(p) for f, p in {**inp, **upstream}.items()}

    DSEd, RAT = feed_ratings(zips)
    nir_st = nir_state_heads(zips['nir_livestock'])
    H_abs, RSE, NAMES, abs_st = abs_livestock(zips['abs_7121_2020_21'])
    GCC = gcc_flags(zips['sa2_2021'])
    A = pd.read_hdf(inp['real_area.h5']).to_numpy().astype(np.float64)
    SN = pd.read_hdf(inp['safe_pur_natl.h5']).to_numpy().astype(np.float64)
    SM = pd.read_hdf(inp['safe_pur_modl.h5']).to_numpy().astype(np.float64)
    ST = np.load(inp['state_id.npy']).astype(np.int64)
    PK = pd.read_hdf(a.pasture).to_numpy().astype(np.float64)
    lu0 = np.load(upstream['class_alloc_2020_21_lumap.npy']).astype(np.int64)
    lm0 = np.load(upstream['class_alloc_2020_21_lmmap.npy']).astype(np.int64)
    cat0 = np.load(upstream['class_alloc_2020_21_category.npy']).astype(np.int64)
    sa2 = fix_state_ties(np.load(upstream['sa2_2021_cells.npy']).astype(np.int64), ST, inp['NLUM_2010-11_mask.tif'])
    assert len(A) == len(lu0) == len(PK) == len(sa2) == NCELL
    HIRF, hir_fy = hir_project_fraction(a.hir_cells, NCELL) if a.hir_cells else (None, None)
    nir_fl = nir_state_feedlot(zips['nir_livestock'])
    fl_area, fl_info = feedlot_area_sa2(zips['alum_2020_21'], inp['NLUM_2010-11_mask.tif'], sa2)
    FL = feedlot_head_sa2(fl_area, nir_fl)
    print('feedlots: NIR state head', nir_fl.round(0).to_dict(), '| NLUM 5.2.2', fl_info, f'{fl_area.sum():,.0f} ha',
          f'in {len(fl_area)} SA2s | estimated head {FL.sum():,.0f}', flush=True)
    o = allocate(DSEd, A, PK, SN, SM, ST, lu0, lm0, cat0, sa2, H_abs, RSE, NAMES.to_dict(), abs_st, nir_st, GCC, HIRF, FL, nir_fl)
    KA, kinfo = out_of_band_k(o, sa2)
    Rr = o['R']
    pl = Rr[Rr.pooled_with_sa3]
    print('typing and k done', kinfo, f'{time.time() - t0:.0f}s', flush=True)
    print(f"town SA2 pairs pooled with their SA3: {len(pl)} ({pl.type.value_counts().to_dict()}); still above the band after "
          f"pooling: {int((pl.flag == 'above').sum())}", flush=True)
    if a.hir_cells:
        print(f"base-year HIR (projects registered up to FY{hir_fy}): {(o['XA'] * A).sum():,.1f} ha adopted on Beef and Sheep "
              f"natural land; head moved onto other cells {o['HR'].head_moved_onto_other_cells.sum():,.1f}; (SA2, type) pairs "
              f"above the band only with HIR {int(o['HR'].above_band_only_with_hir.sum())}", flush=True)

    # ---- write (layout of the 2010-base files)
    LUin = pd.read_hdf(inp['lumap.h5'])
    LMin = pd.read_hdf(inp['lmmap.h5'])
    assert (LUin.index.to_numpy() == np.arange(NCELL)).all() and (LMin.index.to_numpy() == np.arange(NCELL)).all()
    idx = pd.Index(np.arange(NCELL, dtype=np.int64))
    outs = {'lumap.h5': ('lumap', pd.Series(o['lu1'].astype(LUin.dtype), index=LUin.index.copy(), name=LUin.name)),
            'lmmap.h5': ('lmmap', pd.Series(o['lm1'].astype(LMin.dtype), index=LMin.index.copy(), name=LMin.name)),
            'stubble_dse_ha.h5': ('stubble_dse_ha', pd.Series(o['STUB_DSE_HA'].astype(np.float32), index=idx, name='STUBBLE_DSE_HA')),
            'lvstk_k.h5': ('lvstk_k', KA)}
    for f, (key, obj) in outs.items():
        obj.to_hdf(os.path.join(a.out, f), key=key, mode='w', format='table', index=False, complevel=9)
    ncs = [NC.write_like(a.out, f[:-3], obj) for f, (key, obj) in outs.items()]       # NetCDF copies, jinzhu layout
    RAT.to_csv(os.path.join(a.out, 'livestock_dse_ratings.csv'), index=False, lineterminator='\n')
    o['S'].to_csv(os.path.join(a.out, 'livestock_recon_state_type.csv'), index=False, lineterminator='\n')
    o['R'].to_csv(os.path.join(a.out, 'livestock_recon_sa2.csv'), index=False, lineterminator='\n')
    o['HR'].to_csv(os.path.join(a.out, 'livestock_hir_recalibration.csv'), index=False, lineterminator='\n')
    print('written', f'{time.time() - t0:.0f}s', flush=True)

    # ---- provenance (last)
    me = open(os.path.abspath(__file__), 'rb').read()
    P = [['source', k, url, pin, 'sha256 of raw file or API response', note] for k, (url, pin, note) in SOURCES.items()]
    P += [['inherited', f, '', in_sha[f], 'inherited 2010-build input (inherited_2010)', ''] for f in INHERITED]
    P += [['upstream', f, '', in_sha[f], 'input from another 2021-base tool', upstream[f].replace('\\', '/')] for f in upstream]
    P += [['rating', f'DSE_{t}', '', '', repr(DSEd[t]), 'kg DM/head/day, national FY2020-21'] for t in TYPES]
    P += [['constant', 'hir_stocking_factor', '', '', repr(REF['hir_stocking_factor']),
           f'k calibrated with base-year HIR (projects registered up to FY{hir_fy})' if a.hir_cells else 'not applied: no --hir-cells']]
    P += [['script', 'input/Nick_build/scripts/livestock_map.py', '', sha(me), 'sha256 of script', '']]
    P += [['package', k, '', v, 'version', ''] for k, v in sorted(pkg_versions().items())]
    P += [['python', sys.version.split()[0], '', '', 'interpreter', '']]
    for f, (key, _) in outs.items():
        p = os.path.join(a.out, f)
        back = pd.read_hdf(p)
        P.append(['output', f, '', sha_file(p), 'sha256 of file (HDF5 bytes vary between runs)', ''])
        cols = list(back.columns) if isinstance(back, pd.DataFrame) else [back.name]
        P.append(['dataset', f'{f}:/{key} values', '', content_sha(back.to_numpy()), 'sha256 of dtype, shape and array bytes',
                  f'{",".join(map(str, cols))} {back.to_numpy().dtype} {back.shape}'])
        P.append(['dataset', f'{f}:/{key} index', '', content_sha(back.index.to_numpy()), 'sha256 of dtype, shape and array bytes',
                  'cell order'])
    P += [['netcdf', os.path.basename(p), '', h, 'sha256 of NetCDF content (nc_layout.content_sha)', 'checked equal to the .h5 arrays'] for p, h in ncs]
    for f in ('livestock_dse_ratings.csv', 'livestock_recon_state_type.csv', 'livestock_recon_sa2.csv', 'livestock_hir_recalibration.csv'):
        P.append(['output', f, '', sha_file(os.path.join(a.out, f)), 'sha256', ''])
    write_csv(os.path.join(a.out, 'provenance.csv'), ['kind', 'item', 'url', 'sha256', 'note', 'source'], P)
    print('done', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
