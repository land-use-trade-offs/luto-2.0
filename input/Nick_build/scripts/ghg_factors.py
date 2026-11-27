#!/usr/bin/env python
"""Build LUTO's 2021-base agricultural GHG emission factors (livestock per head, off-land, crop and irrigated-pasture SOIL,
crop residues, field burning and rice CH4).

Sources (public, downloaded at run time, sha256-checked against pinned values):
  - Australia's National Greenhouse Accounts API (NIR 2024), StatesOutputTimeSeries: FY2020-21 emissions by state of the
    livestock, off-land (swine, poultry), fertiliser and crop-field nodes in REF. The API is a changing source: 827 raw
    responses are cached and pinned one by one.
  - DCCEEW NIR 2024 activity table, agriculture livestock (state populations by class, FY2020-21).
  - DCCEEW NIR 2024 activity table, agriculture fertiliser (N mass by state and system; reported rates only, no factor uses it).
  - NIR 2024 Volume 2, Table A5.5.8.1 (crop production areas by state and fertiliser system): the FY2020-21 rows are a
    hand-entered reference table (REF), each with its PDF page and the PDF sha256; the tool does not read the PDF.
Inputs from the other 2021-base tools: sa2_2021_cells.npy (nlum_map.py); the final lumap.h5 and lmmap.h5
(livestock_map.py); x_mrj.npy (eligibility.py). Inherited 2010-build inputs from --input (inherited_2010/, copies of 2010-build files),
each sha256 recorded in the provenance file: agGHG_lvstk.h5, agGHG_crops.h5, agGHG_irrpast.h5, agGHG_lvstk_off_land.csv (the
file layouts, the NaN masks, every unchanged field and the energy and lifecycle values), demand_projections.h5 (FY2021
off-land production), real_area.h5, ag_landuses.csv, state_id.npy and NLUM_2010-11_mask.tif. The outputs are therefore not
built from public sources alone.

Method:
  1. State per cell: the SA2 2021 key with state ties fixed (as livestock_map.py); first digit.
  2. agGHG_lvstk.h5: ENTERIC, MANURE_MGT, DUNG_URINE, IND_LEACH_RUNOFF per head = NIR FY2021 state emissions of the mapped
     nodes (AR5 CO2e) / NIR state population of the type; the national value where the state has no population. Cells NaN in
     the 2010 file stay NaN; the other 12 fields are unchanged.
  3. agGHG_lvstk_off_land.csv: pork, chicken and eggs enteric and manure rows = NIR FY2021 national emissions / LUTO's FY2021
     default-scenario production (eggs at 60 g); energy rows and aquaculture unchanged.
  4. Fertiliser N2O (direct + atmospheric deposition + leaching) by state and NIR system, FY2021.
     Base: per NIR hectare (Table A5.5.8.1) by state, the national rate where the state rate is not positive, on the
     cells finite in the 2010 file. Crops: per hectare of LUTO cropping land on the final map by system and state
     (NSW and ACT pooled as NIR reports them), the national rate where a state has none, on every cell eligible for the crop in
     x_mrj; other cells keep that base. Irrigated pasture: one national rate = NIR irrigated-pasture fertiliser
     N2O / LUTO irrigated-pasture area on the final map, on every cell eligible for an irrigated livestock land use.
  5. INTERIM FILL: blank energy and lifecycle fields of agGHG_crops.h5 and agGHG_irrpast.h5 on eligible cells
     take the area-weighted state mean of the (field, lm, lu) over cells with values (national mean where the state has none).
     Interim, to be rebuilt; these fields are reported only, off every constraint row.
  6. Crop fields CO2E_KG_HA_RESIDUES (3.D.a.4 and 3.D.b.2.v crop leaves), CO2E_KG_HA_BURNING (3.F) and CO2E_KG_HA_RICE_CH4
     (3.C): NIR FY2021 emissions of the crop leaves by crop group and state / LUTO cropping land of the group and state on
     the final map (NSW and ACT pooled), the group's national rate where a state has no LUTO area; on every cell eligible
     for the (lm, crop) in x_mrj, NaN elsewhere; crops with no NIR leaf (perennial horticulture) 0. Pasture residue
     leaves, urea and liming stay in the exogenous series. Columns for the (lm, crop) pairs of
     the 2010 layout and every other pair eligible in x_mrj, appended after the 2010 layout.
Outputs in --out: agGHG_lvstk.h5, agGHG_crops.h5, agGHG_irrpast.h5 (layouts of the 2010-base files; agGHG_crops.h5 with the
three crop fields appended), agGHG_lvstk_off_land.csv (text layout of the 2010-base file), ghg_crop_rates.csv (crop and
irrigated-pasture rates by system and state), ghg_crop_field_rates.csv (crop-field rates by field, group and state),
ghg_crop_field_placed.csv (crop-field tonnes placed on the final map against NIR by field and state),
interim_fill_energy_lifecycle.csv (every interim fill); NetCDF copies in the jinzhu layout (nc_layout.py) of
agGHG_lvstk (AGGHG_LVSTK over (column, cell), levels animal and field) and agGHG_irrpast (one variable per column); the
copy of agGHG_crops is written by crop_economics.py with the final file; provenance.csv (last).

Run as a plain script from the repository root (no luto imports):

    python input/Nick_build/scripts/ghg_factors.py --input input/Nick_build/inherited_2010 --nlum-map input/Nick_build/nlum_map --map-dir input/Nick_build/livestock_map
        --x-mrj input/Nick_build/eligibility/x_mrj.npy --out input/Nick_build/ghg_factors --cache input/Nick_build/_cache [--offline] [--live-check] [--source NAME=PATH]
        [--source-dir DIR]

Dependencies: numpy, pandas, PyTables, openpyxl, rasterio, scipy, xarray, netCDF4. No luto imports (nc_layout.py is a
sibling module).
Memory: peak resident memory 73.2 GiB on the full grid with the crop fields (measured 19 October 2026 in a full input build:
resident set size of the process and its children, sampled every 0.5 s with psutil on Windows; run time 24 minutes); run
on a machine with at least 128 GB of RAM.
Output values are deterministic; HDF5 file bytes carry a group timestamp, so provenance records dataset content hashes.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
from importlib import metadata

import numpy as np
import pandas as pd
import rasterio

import nc_layout as NC

# ----------------------------------------------------------------------------- sources and pins
ANGA = ('https://greenhouseaccounts.climatechange.gov.au/api/StatesOutputTimeSeries?sectorid={s}&urlencodedlocationlist={l}'
        '&fuelid=1&gasid={g}')
SOURCES = {   # name: (url, sha256 of the raw file or response, note); the API responses are added from ANGA_PINS below
    'nir_livestock': ('https://www.dcceew.gov.au/sites/default/files/documents/'
                      'activity-table-1990-2024-agriculture-livestock-national.xlsx',
                      '669ac799188e97f993dc85662886856295fe2dc0e68e26590058d765e97efa59',
                      'DCCEEW NIR 2024 activity table, agriculture livestock, CC BY 4.0'),
    'nir_fertiliser': ('https://www.dcceew.gov.au/sites/default/files/documents/'
                       'activity-table-1990-2024-agriculture-fertiliser.xlsx',
                       'd7182ada5900e915e70573c3e9d61b8d81b6e3adfe902e56fbe1deec79e4d1ff',
                       'DCCEEW NIR 2024 activity table, agriculture fertiliser, CC BY 4.0'),
}
INHERITED = ['agGHG_lvstk.h5', 'agGHG_crops.h5', 'agGHG_irrpast.h5', 'agGHG_lvstk_off_land.csv', 'demand_projections.h5',
             'real_area.h5', 'ag_landuses.csv', 'state_id.npy', 'NLUM_2010-11_mask.tif']
NCELL = 6956407
FY = '2020-21'
LOC = {3: 'AUSTRALIA', 6: 'NSW', 11: 'VIC', 8: 'QLD', 9: 'SA', 12: 'WA', 10: 'TAS', 7: 'NT', 5: 'ACT'}
STS = ['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT']
ST_NAME = {1: 'NSW', 2: 'VIC', 3: 'QLD', 4: 'SA', 5: 'WA', 6: 'TAS', 7: 'NT', 8: 'ACT', 9: 'OT'}

# ============================================================================= reference table (hand-entered)
# Each value with its source. Changing any is a ruling.
NIR_VOL2 = ('https://www.dcceew.gov.au/sites/default/files/documents/national-inventory-report-2024-vol-2.pdf',
            '9d2eab75b733fd93f6497529beb73304be4957a67b6b7adb247d419d6d1dd6c7')
REF = {
    # LUTO field <- NIR 2024 nodes (ANGA sector ids), AR5 CO2e (gas 103), summed (mapping inferred)
    'lvstk_nodes': {
        'DAIRY': {'ENTERIC': [684], 'MANURE_MGT': [737, 5557, 5562, 5555], 'DUNG_URINE': [5835, 5837],
                  'IND_LEACH_RUNOFF': [5851, 5855]},
        'BEEF': {'ENTERIC': [5811], 'MANURE_MGT': [5823], 'DUNG_URINE': [5836, 5838], 'IND_LEACH_RUNOFF': [5852, 5856]},
        'SHEEP': {'ENTERIC': [705], 'MANURE_MGT': [945], 'DUNG_URINE': [5321, 5340], 'IND_LEACH_RUNOFF': [5377, 5409]}},
    # NIR activity table group of each LUTO type (populations summed over classes)
    'lvstk_group': {'DAIRY': 'Number of dairy cattle', 'BEEF': 'Number of beef cattle - pasture', 'SHEEP': 'Number of sheep'},
    # off-land, national: swine CH4 enteric 720, manure 1092; swine N2O 1092 + 3.D nodes; poultry layers 5250,
    # meat chickens 5251; the other poultry N2O nodes split by the layers / meat share of 3.B.4.g manure N2O
    'swine_ch4_enteric': 720, 'swine_ch4_manure': 1092, 'swine_n2o': [1092, 5263, 5279, 5300, 5322, 5341, 5378, 5410],
    'layers': 5250, 'meat_chickens': 5251, 'poultry_other_n2o': [5559, 5563, 5307, 5348, 5385, 5417],
    # LUTO default scenario keys of demand_projections.h5 (luto settings at 96455591); egg mass 60 g
    'demand_keys': ('SSP2', 'BAU', 'BAU', 2050, 'Trend', 1, 'BAU'), 'egg_g': 60,
    # fertiliser N2O nodes per NIR system: (direct 3.D.a.1, deposition 3.D.b.1.i, leaching 3.D.b.2.i, activity-table label)
    'fert_systems': {'irr_pasture': (5285, 5364, 5396, 'Irrigated pasture'), 'irr_crop': (5286, 5365, 5397, 'Irrigated crops'),
                     'dry_pasture': (5287, 5366, 5398, 'Non-irrigated pasture'),
                     'dry_crop': (5288, 5367, 5399, 'Non-irrigated crops'), 'sugar': (5289, 5368, 5400, 'Sugar'),
                     'cotton': (5290, 5369, 5401, 'Cotton'), 'hort': (5291, 5370, 5402, 'Horticultural vegetable crops')},
    'hort_lus': ['Vegetables', 'Apples', 'Citrus', 'Grapes', 'Nuts', 'Pears', 'Plantation fruit', 'Stone fruit',
                 'Tropical stone fruit'],
    # NIR 2024 Vol 2 Table A5.5.8.1, crop production areas by state (ha), row 2021 (= FY2020-21; label = FY end year).
    # PDF pages (1-based) of NIR_VOL2: NSW/ACT and NT p. 200, QLD and SA p. 201, TAS and
    # VIC p. 202, WA p. 203. Printed '-' is 0. Columns: irr_pasture, irr_crop, dry_pasture, dry_crop, sugar, cotton, hort.
    'nir_area_2021': {
        'NSW': ([160332, 275777, 3862270, 7026545, 12863, 192760, 111760], 200),   # block 'New South Wales/ACT'
        'NT': ([1285, 415, 40544, 3973, 0, 0, 6295], 200),
        'QLD': ([61530, 59272, 3149039, 1909060, 328978, 104620, 105767], 201),
        'SA': ([57165, 12737, 2378602, 3651157, 0, 0, 90856], 201),
        'TAS': ([76468, 7531, 773793, 10344, 0, 0, 20277], 202),
        'VIC': ([61581, 12585, 3971377, 3590351, 0, 0, 127602], 202),
        'WA': ([14131, 17676, 5070516, 8309931, 0, 0, 20763], 203)},
    # PDF text-layer typos in the same table (row 2019, not used by any factor here; listed so the table is complete)
    'nir_area_typos': [('TAS', 'dry_crop', 2019, 202, '279,0844', 279084), ('TAS', 'hort', 2019, 202, '17,4623', 17462)],
    # Crop fields (rice CH4, field burning and crop residues are modelled crop fields; urea, liming and pasture residues
    # stay in the exogenous series): every crop leaf of the field's NIR nodes, by LUTO crop group (mapping inferred).
    # Roots: residues 3.D.a.4 Crop Residue (direct N2O) and 3.D.b.2.v Crop Residues (leaching N2O); burning 3.F Field
    # Burning of Agricultural Residues; rice 3.C Rice Cultivation. Leaves with no FY2020-21 emissions are listed too.
    'crop_field_leaves': {
        'CO2E_KG_HA_RESIDUES': {
            'winter_cereals': [1289, 1298, 5353, 1295, 5839, 5424, 5426, 5429, 5433, 5863],   # barley, wheat, oats, triticale, other
            'summer_cereals': [1290, 1294, 5425, 5432],                                     # maize, sorghum
            'rice': [5354, 5430], 'legumes': [1299, 5840, 5434, 5437, 5864],                # pulse, peanuts
            'oilseeds': [5842, 5866], 'sugar': [1301, 5436], 'cotton': [5841, 5865], 'tuber_root': [1300, 5435],
            'forage': [5844, 5868], 'hops': [5843, 5867]},
        'CO2E_KG_HA_BURNING': {
            'winter_cereals': [1371, 1370, 5443, 1379, 5854], 'summer_cereals': [1372, 1378], 'rice': [5444],
            'legumes': [1380, 5859], 'oilseeds': [5846], 'sugar': [1382], 'cotton': [5860], 'tuber_root': [1381],
            'forage': [5862], 'hops': [5861]},
        'CO2E_KG_HA_RICE_CH4': {'rice': [1238]}},                                           # continuously flooded
    # pasture leaves of the residue nodes (lucerne, other legume pasture, grass clover, perennial pasture, annual
    # grasses): exogenous, not a crop field
    'crop_field_pasture_leaves': [5845, 5847, 5848, 5849, 5850, 5870, 5871, 5872, 5873, 5874],
    # LUTO land uses of each crop group; crops in no group (perennial horticulture) have no NIR leaf and take 0
    'crop_field_groups': {'winter_cereals': ['Winter cereals'], 'summer_cereals': ['Summer cereals'], 'rice': ['Rice'],
                          'legumes': ['Winter legumes', 'Summer legumes'], 'oilseeds': ['Winter oilseeds', 'Summer oilseeds'],
                          'sugar': ['Sugar'], 'cotton': ['Cotton'], 'tuber_root': ['Vegetables'], 'forage': ['Hay'],
                          'hops': ['Other non-cereal crops']},
}
REF['interim_label'] = 'INTERIM FILL: to be rebuilt; reported only, off every constraint row'
# ============================================================================= end of reference table
SYSTEMS = list(REF['fert_systems'])
LV = ('Beef', 'Sheep', 'Dairy')


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
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (luto-ghg-factors/1.0)',
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


# ----------------------------------------------------------------------------- source tables
def anga_frame(paths):
    """Long table of the pinned API responses (one row per location, sector, gas and year, in that order)."""
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


def nir_populations(xlsx):
    """NIR activity table, FY2020-21 population by sheet (state), group and class."""
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
                if fy == FY:
                    rows.append(dict(state=st.upper(), group=grp, cls=cls.strip(), value=r[i]))
    return pd.DataFrame(rows)


def nir_fert_n(xlsx):
    """NIR activity table: fertiliser N (Gg N) FY2020-21 by (state, system label)."""
    import openpyxl
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    R = list(wb['Agriculture - Fertilisers'].iter_rows(values_only=True))
    c21 = R[5].index(FY)
    NM, st = {}, None
    for r in R[6:70]:
        if r[0]:
            st = r[0].strip().upper()
            continue
        if r[1]:
            NM[(st, r[1])] = r[c21]
    return NM


def nir_area_table():
    rows = [dict(state=s, system=k, area_ha=float(v)) for s, (vals, _) in REF['nir_area_2021'].items()
            for k, v in zip(SYSTEMS, vals)]
    return pd.DataFrame(rows)


def fix_state_ties(sa2, ST, mask_path):
    """As livestock_map.py: a cell whose SA2 state digit disagrees with state_id takes the SA2 of the
    nearest cell (grid distance) whose SA2 digit equals that state_id and agrees with its own state_id."""
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


# ----------------------------------------------------------------------------- step 2: livestock per head
def lvstk_factors(A, P):
    """kg CO2e per head by (type, field) and state (index STS + AUSTRALIA); NaN where NIR has no population."""
    a = A[(A.gas_id == 103) & (A.fy_label == FY) & A.loc_name.isin(STS + ['AUSTRALIA'])]
    out = {}
    for an, flds in REF['lvstk_nodes'].items():
        p = P[P.group == REF['lvstk_group'][an]].groupby('state').value.sum().reindex(STS + ['AUSTRALIA']).fillna(0.)
        for f, nodes in flds.items():
            e = a[a.sector_id.isin(nodes)].groupby('loc_name').value.sum().reindex(STS + ['AUSTRALIA']).fillna(0.)
            out[(an, 'CO2E_KG_HEAD_' + f)] = pd.Series({s: (e[s] * 1e6 / p[s]) if p[s] > 0 else np.nan for s in STS + ['AUSTRALIA']})
    return out


def lvstk_table(lv, F, st):
    new = lv.copy()
    for c in lv.columns:
        if c not in F:
            continue
        v = F[c]
        val = pd.Series(st).map(v).to_numpy(dtype=float).copy()
        val[~np.isfinite(val)] = v['AUSTRALIA']
        old = lv[c].to_numpy()
        new[c] = np.where(~np.isfinite(old), np.nan, val).astype(np.float32)
    new.columns = pd.MultiIndex.from_tuples([tuple(c) for c in new.columns])    # no level names
    return new


# ----------------------------------------------------------------------------- step 3: off-land
def offland(A, dd, orig_path):
    a = A[(A.fy_label == FY) & (A.loc_name == 'AUSTRALIA')]

    def g(nodes, gas):
        return a[a.sector_id.isin(nodes) & (a.gas_id == gas)].value.sum()   # Gg
    D = dd.loc[REF['demand_keys']].copy()
    D.loc['eggs'] = D.loc['eggs'] * REF['egg_g'] / 1000 / 1000
    P21 = D.loc[['pork', 'chicken', 'eggs'], 'PRODUCTION'][2021]
    sw_ch4_ent, sw_ch4_mm = g([REF['swine_ch4_enteric']], 108), g([REF['swine_ch4_manure']], 108)
    sw_n2o = sum(g([n], 109) for n in REF['swine_n2o'])
    lay_ch4, meat_ch4 = g([REF['layers']], 108), g([REF['meat_chickens']], 108)
    lay_n2o_mm, meat_n2o_mm = g([REF['layers']], 109), g([REF['meat_chickens']], 109)
    sh = lay_n2o_mm / (lay_n2o_mm + meat_n2o_mm)
    po_other = sum(g([n], 109) for n in REF['poultry_other_n2o'])
    lay_n2o, meat_n2o = lay_n2o_mm + sh * po_other, meat_n2o_mm + (1 - sh) * po_other
    NEW = {('pork', 'Enteric fermentation (CH4)'): sw_ch4_ent, ('pork', 'Manure (CH4)'): sw_ch4_mm, ('pork', 'Manure (N2O)'): sw_n2o,
           ('eggs', 'Manure (CH4)'): lay_ch4, ('eggs', 'Manure (N2O)'): lay_n2o, ('chicken', 'Manure (CH4)'): meat_ch4,
           ('chicken', 'Manure (N2O)'): meat_n2o}
    C = pd.read_csv(orig_path, thousands=',')
    oc = C.copy().astype({'Production [ t ]': 'float64', 'Emissions [ t CO2eq ]': 'float64'})
    for i, r in C.iterrows():
        k = (r.COMMODITY, r['Emission Source'])
        if k in NEW:
            t = float(P21[r.COMMODITY])
            e = NEW[k] * 1000   # Gg -> t
            oc.loc[i, 'Production [ t ]'] = t
            oc.loc[i, 'Emissions [ t CO2eq ]'] = e
            oc.loc[i, 'Emission Intensity [ kg CO2eq / kg ]'] = e / t
    # This table goes through a CSV write and a read back with pandas' default float parser,
    # which moves some intensities by 1 ulp; the same round trip is kept so the file reproduces the shipped bytes.
    oc = pd.read_csv(io.StringIO(oc.to_csv(index=False)))
    # same columns and text formatting as the 2010-base file; non-energy, non-aquaculture rows replaced
    orig = pd.read_csv(orig_path)
    out = orig.copy()
    for i in range(len(orig)):
        if (orig.loc[i, 'Emission Source'] in ('Enteric fermentation (CH4)', 'Manure (CH4)', 'Manure (N2O)')
                and orig.loc[i, 'COMMODITY'] != 'aquaculture'):
            out.loc[i, 'Production [ t ]'] = f"{oc.loc[i, 'Production [ t ]']:,.0f}"
            out.loc[i, 'Emissions [ t CO2eq ]'] = f"{oc.loc[i, 'Emissions [ t CO2eq ]']:,.0f}"
            out.loc[i, 'Emission Intensity [ kg CO2eq / kg ]'] = oc.loc[i, 'Emission Intensity [ kg CO2eq / kg ]']
    return out


# ----------------------------------------------------------------------------- step 4: fertiliser N2O rates
def fert_systems(A, NM):
    """NIR FY2021 fertiliser N2O (t CO2e, direct + deposition + leaching) and N (Gg) by system and state."""
    a = A[(A.gas_id == 103) & (A.fy_label == FY) & A.loc_name.isin(STS)]

    def g(nodes):
        return a[a.sector_id.isin(nodes)].groupby('loc_name').value.sum().reindex(STS).fillna(0.) * 1000   # t CO2e
    rows = []
    for name, (d, dep, le, lab) in REF['fert_systems'].items():
        ed, edep, ele = g([d]), g([dep]), g([le])
        for s in STS:
            rows.append(dict(system=name, state=s, n_gg=NM.get((s, lab), 0) or 0., fert_direct_t=ed[s], fert_dep_t=edep[s],
                             fert_leach_t=ele[s]))
    S = pd.DataFrame(rows)
    S['fert_n2o_t'] = S.fert_direct_t + S.fert_dep_t + S.fert_leach_t
    return S


def crop_system(lu, lm):
    if lu == 'Sugar':
        return 'sugar'
    if lu == 'Cotton':
        return 'cotton'
    if lu in REF['hort_lus']:
        return 'hort'
    return 'irr_crop' if lm == 'irr' else 'dry_crop'


def nir_ha_rates(S, AR):
    """NIR-hectare base: rate per NIR hectare by (system, state), ACT on the NSW/ACT area; national rate per system."""
    AR = pd.concat([AR, AR[AR.state == 'NSW'].assign(state='ACT')])
    M = S.merge(AR.rename(columns={'area_ha': 'nir_area_ha'}), on=['state', 'system'], how='left')
    M['fert_n2o_kg_ha_nir'] = np.where(M.nir_area_ha > 0, M['fert_n2o_t'] * 1000 / M.nir_area_ha, np.nan)
    nat = {s_: d.fert_n2o_t.sum() * 1000 / d.nir_area_ha[d.state != 'ACT'].sum() for s_, d in M.groupby('system')}
    return M.set_index(['system', 'state']).fert_n2o_kg_ha_nir, nat


def luto_ha_rates(S, AR, lumap, lmmap, area, st_pool, LUS):
    """Crop rates per hectare of LUTO cropping land on the final map by system and state (NSW and
    ACT pooled); irrigated pasture per NIR hectare (superseded on eligible cells by the irrigated pasture rate)."""
    def system(lu, lm):
        if any(a in lu for a in LV):
            return 'irr_pasture' if lm == 'irr' else None
        if 'Unallocated' in lu:
            return None
        return crop_system(lu, lm)
    m = lumap >= 0
    code = lumap * 2 + (lmmap == 1)
    look = {j * 2 + k: system(LUS[j], 'irr' if k == 1 else 'dry') for j in range(len(LUS)) for k in (0, 1)}
    sysv = pd.Series(code[m]).map(look).to_numpy(dtype=object)
    D = pd.DataFrame(dict(system=sysv, state=st_pool[m], area=area[m]))
    LA = D.dropna().groupby(['system', 'state']).area.sum().rename('luto_area_ha')
    E = S[S.state != 'ACT'][['system', 'state', 'n_gg', 'fert_n2o_t']]
    AR = AR.rename(columns={'area_ha': 'nir_area_ha'})
    T = E.merge(LA.reset_index(), on=['system', 'state'], how='outer').merge(AR, on=['system', 'state'], how='left').fillna(
        {'luto_area_ha': 0., 'fert_n2o_t': 0., 'n_gg': 0.})
    crop = T.system.isin(['irr_crop', 'dry_crop', 'sugar', 'cotton', 'hort'])
    T['basis'] = np.where(crop, 'LUTO cropping land (final map)', np.where(T.system == 'irr_pasture',
                                                                            'NIR irrigated-pasture area', 'not modelled'))
    T['denom_ha'] = np.where(crop, T.luto_area_ha, np.where(T.system == 'irr_pasture', T.nir_area_ha, np.nan))
    T['rate_kg_co2e_ha'] = np.where(T.denom_ha > 0, T.fert_n2o_t * 1000 / T.denom_ha, np.nan)
    T['n_rate_per_cropping_ha'] = np.where(T.luto_area_ha > 0, T.n_gg * 1e6 / T.luto_area_ha, np.nan)
    T['n_rate_per_nir_ha'] = np.where(T.nir_area_ha > 0, T.n_gg * 1e6 / T.nir_area_ha, np.nan)
    T['sown_share_nir'] = np.where(T.luto_area_ha > 0, T.nir_area_ha / T.luto_area_ha, np.nan)
    nat = {}
    for s_, g in T[T.system != 'dry_pasture'].groupby('system'):
        den = g.denom_ha.sum()
        nat[s_] = g.fert_n2o_t.sum() * 1000 / den if den > 0 else np.nan
    T['national_rate_kg_co2e_ha'] = T.system.map(nat)
    T['flag'] = np.select([crop & (T.luto_area_ha > 0) & (T.fert_n2o_t == 0), crop & (T.luto_area_ha == 0) & (T.fert_n2o_t > 0)],
                          ['LUTO cropping land, NIR has no fertiliser N2O (rate 0)', 'NIR emissions with no LUTO cropping land'], '')
    return T


# ----------------------------------------------------------------------------- steps 4 and 5: per-cell SOIL and interim fill
def soil_tables(cr10, ip10, S, AR, T, st, x, lumap, lmmap, area, LUS):
    # NIR-hectare base: per NIR ha, cells finite in the 2010 file; non-positive state rates take the national rate
    rate4, nat4 = nir_ha_rates(S, AR)

    def cell_rate4(sysn):
        r = pd.Series(st).map({s: rate4.get((sysn, s), np.nan) for s in set(st)}).to_numpy(dtype=float).copy()
        r[~(r > 0)] = nat4[sysn]
        return r
    cols = [tuple(c) for c in cr10.columns]
    assert (cr10.dtypes == np.float32).all()
    V = np.array(cr10.to_numpy(dtype=np.float32), order='F', copy=True)   # column work on one array (wide DataFrame setitem is slow)
    for i, c in enumerate(cols):
        if c[0] == 'CO2E_KG_HA_SOIL':
            old = V[:, i].copy()
            V[:, i] = np.where(np.isfinite(old), cell_rate4(crop_system(c[2], c[1])), np.nan).astype(np.float32)
    ip = ip10.copy()
    old = ip10['CO2E_KG_HA_SOIL'].to_numpy()
    ip['CO2E_KG_HA_SOIL'] = np.where(np.isfinite(old), cell_rate4('irr_pasture'), np.nan).astype(old.dtype)

    # crops on eligible cells: per LUTO cropping ha; NaN state rates take the national rate
    rate = {(r.system, r.state): r.rate_kg_co2e_ha for r in T.itertuples()}
    natT = T.groupby('system').national_rate_kg_co2e_ha.first().to_dict()
    st_pool = np.where(st == 'ACT', 'NSW', st)
    ust = sorted(set(st_pool))

    def cell_rate(sysn):
        vals = {s: rate.get((sysn, s), np.nan) for s in ust}
        vals = {s: (v if np.isfinite(v) else natT[sysn]) for s, v in vals.items()}
        return pd.Series(st_pool).map(vals).to_numpy(dtype=np.float64)
    for i, c in enumerate(cols):
        if c[0] != 'CO2E_KG_HA_SOIL':
            continue
        m, j = (1 if c[1] == 'irr' else 0), LUS.index(c[2])
        elig = np.asarray(x[m, j])
        V[:, i] = np.where(elig, cell_rate(crop_system(c[2], c[1])), V[:, i].astype(np.float64)).astype(np.float32)

    # irrigated pasture: one national rate per LUTO irrigated-pasture ha, on cells eligible for irrigated livestock
    lv_j = [LUS.index(l) for l in LUS if any(a in l for a in LV)]
    elig_irr = np.zeros(len(lumap), bool)
    for j in lv_j:
        elig_irr |= np.asarray(x[1, j])
    ipn = S[S.system == 'irr_pasture']
    alloc_irr = np.isin(lumap, lv_j) & (lmmap == 1)
    r219 = dict(nir_national_fert_n2o_t=ipn.fert_n2o_t.sum(), nir_national_n_gg=ipn.n_gg.sum(), luto_irr_pasture_ha=area[alloc_irr].sum())
    r219['rate_kg_co2e_ha'] = r219['nir_national_fert_n2o_t'] * 1000 / r219['luto_irr_pasture_ha']
    r219['n_rate_kg_n_per_luto_ha'] = r219['nir_national_n_gg'] * 1e6 / r219['luto_irr_pasture_ha']
    dt = ip['CO2E_KG_HA_SOIL'].dtype
    ip['CO2E_KG_HA_SOIL'] = np.where(elig_irr, r219['rate_kg_co2e_ha'], ip['CO2E_KG_HA_SOIL'].to_numpy()).astype(dt)

    # INTERIM FILL: blank energy and lifecycle fields on eligible cells, area-weighted state mean
    ust = sorted(set(st))
    SM = {s: st == s for s in ust}
    fills = []

    def fill(v, blank, ok):
        nat = np.average(v[ok], weights=area[ok]) if ok.any() else np.nan
        smean = {s: (np.average(v[ok & SM[s]], weights=area[ok & SM[s]]) if (ok & SM[s]).any() else nat) for s in ust}
        fv = np.zeros(len(v))
        for s in ust:
            fv[SM[s]] = smean[s]
        return np.where(blank, fv, v), smean, nat
    for i, c in enumerate(cols):
        if c[0] == 'CO2E_KG_HA_SOIL':
            continue
        m, j = (1 if c[1] == 'irr' else 0), LUS.index(c[2])
        elig = np.asarray(x[m, j])
        v = V[:, i].astype(np.float64)
        ok = np.isfinite(v)
        blank = elig & ~ok
        if not blank.any():
            continue
        newv, smean, nat = fill(v, blank, ok)
        V[:, i] = newv.astype(np.float32)
        for s in ust:
            b = blank & SM[s]
            if b.any():
                fills.append(dict(file='agGHG_crops', field=c[0], lm=c[1], lu=c[2], state=s, filled_cells=int(b.sum()),
                                  filled_ha=area[b].sum(), value=smean[s],
                                  source='state mean' if (ok & SM[s]).any() else ('national mean' if np.isfinite(nat) else 'none (left NaN)'),
                                  still_nan_cells=int((b & ~np.isfinite(newv)).sum())))
    for c in [c for c in ip.columns if 'CO2E' in c and c != 'CO2E_KG_HA_SOIL']:
        v = ip[c].to_numpy().astype(np.float64)
        ok = np.isfinite(v)
        blank = elig_irr & ~ok
        if not blank.any():
            continue
        newv, smean, nat = fill(v, blank, ok)
        ip[c] = newv.astype(ip[c].dtype)
        for s in ust:
            b = blank & SM[s]
            if b.any():
                fills.append(dict(file='agGHG_irrpast', field=c, lm='irr', lu='irrigated livestock', state=s, filled_cells=int(b.sum()),
                                  filled_ha=area[b].sum(), value=smean[s], source='state mean' if (ok & SM[s]).any() else 'national mean',
                                  still_nan_cells=int((b & ~np.isfinite(newv)).sum())))
    FL = pd.DataFrame(fills, columns=['file', 'field', 'lm', 'lu', 'state', 'filled_cells', 'filled_ha', 'value', 'source',
                                      'still_nan_cells'])
    FL['label'] = REF['interim_label']
    cr = pd.DataFrame(V, index=cr10.index.copy(), columns=pd.MultiIndex.from_tuples(cols))   # no level names
    return cr, ip, FL, r219


# ----------------------------------------------------------------------------- step 6: crop fields
def crop_fields(A, st, x, lumap, lmmap, area, LUS, pairs):
    """CO2E_KG_HA_RESIDUES, _BURNING and _RICE_CH4 per hectare of cropping land: NIR FY2020-21 emissions of the field's
    crop leaves by crop group and state / LUTO cropping land of the group and state on the final map (both land
    managements; NSW and ACT pooled). A state with no rate takes the group's national rate; a state with LUTO area and
    no NIR emissions 0. Filled on every cell eligible for the (lm, crop) in x_mrj, NaN elsewhere; crops in no group 0.
    Returns the new columns, the rate table and the placed tonnes against NIR by field and state."""
    FL, GR = REF['crop_field_leaves'], REF['crop_field_groups']
    rows = [(f, g, l) for f, gs in FL.items() for g, ls in gs.items() for l in ls]
    leaf = pd.DataFrame(rows, columns=['field', 'group', 'sector_id'])
    d = A[(A.gas_id == 103) & (A.fy_label == FY)]
    d = leaf.merge(d[['sector_id', 'loc_name', 'value']], on='sector_id').assign(t=lambda z: z.value * 1000.0)
    nat_chk = d[d.loc_name != 'AUSTRALIA'].groupby('sector_id').t.sum().sub(d[d.loc_name == 'AUSTRALIA'].groupby('sector_id').t.sum(),
                                                                            fill_value=0.0)
    assert nat_chk.abs().max() < 1e-6, nat_chk.abs().max()                 # states sum to national
    Lst = d[d.loc_name != 'AUSTRALIA'].assign(state=lambda z: z.loc_name.replace({'ACT': 'NSW'}))   # NSW + ACT, once
    st_pool = np.where(st == 'ACT', 'NSW', st)
    LU2G = {lu: g for g, lus in GR.items() for lu in lus}
    grp_cell = np.array([LU2G.get(LUS[j]) if j >= 0 else None for j in lumap], dtype=object)
    m = grp_cell != None  # noqa: E711
    LA = pd.DataFrame(dict(group=grp_cell[m], state=st_pool[m], area=area[m])).groupby(['group', 'state']).area.sum().rename('luto_area_ha')
    E = Lst.groupby(['field', 'group', 'state']).t.sum().rename('nir_t').reset_index()
    states = sorted(set(st_pool))
    grid = pd.MultiIndex.from_product([list(FL), list(GR), states], names=['field', 'group', 'state']).to_frame(index=False)
    grid = grid[[g in FL[f] for f, g in zip(grid.field, grid.group)]]
    T = grid.merge(E, how='left').merge(LA.reset_index(), how='left').fillna({'nir_t': 0.0, 'luto_area_ha': 0.0})
    T['rate_kg_co2e_ha'] = np.where(T.luto_area_ha > 0, T.nir_t * 1000 / T.luto_area_ha, np.nan)
    nat = {(f, g): (z.nir_t.sum() * 1000 / z.luto_area_ha.sum() if z.luto_area_ha.sum() > 0 else 0.0) for (f, g), z in T.groupby(['field', 'group'])}
    T['national_rate_kg_co2e_ha'] = [nat[(f, g)] for f, g in zip(T.field, T.group)]
    T['flag'] = np.select([(T.luto_area_ha > 0) & (T.nir_t == 0), (T.luto_area_ha == 0) & (T.nir_t > 0)],
                          ['LUTO cropping land, NIR has none (rate 0)', 'NIR emissions with no LUTO cropping land (not placed)'], '')
    rate = {(r.field, r.group, r.state): r.rate_kg_co2e_ha for r in T.itertuples()}
    new, placed = {}, []
    for field in FL:
        for lm, lu in pairs:
            mi, j = (1 if lm == 'irr' else 0), LUS.index(lu)
            g = LU2G.get(lu) if LU2G.get(lu) in FL[field] else None
            if g is None:
                cell = np.zeros(len(st_pool))                               # no NIR leaf for this crop: rate 0
            else:
                vals = {s: rate.get((field, g, s), np.nan) for s in states}
                vals = {s: (v if np.isfinite(v) else nat[(field, g)]) for s, v in vals.items()}
                cell = pd.Series(st_pool).map(vals).to_numpy(dtype=np.float64)
            new[(field, lm, lu)] = np.where(np.asarray(x[mi, j]), cell, np.nan).astype(np.float32)
    for field in FL:                                                        # placed on the final map against NIR
        for s in states:
            t_luto = 0.0
            for lm, lu in pairs:
                mi, j = (1 if lm == 'irr' else 0), LUS.index(lu)
                al = (lumap == j) & (lmmap == mi) & (st_pool == s)
                t_luto += float(np.nansum(new[(field, lm, lu)][al].astype(np.float64) * area[al]) / 1000)
            t_nir = float(E[(E.field == field) & (E.state == s)].nir_t.sum())
            unpl = float(T[(T.field == field) & (T.state == s) & T.flag.str.startswith('NIR emissions')].nir_t.sum())
            placed.append(dict(field=field, state=s, luto_t=t_luto, nir_crop_t=t_nir, unplaced_t=unpl, residual_t=t_luto - t_nir + unpl))
    P = pd.DataFrame(placed)
    assert P.residual_t.abs().max() < 0.05, P.residual_t.abs().max()         # every NIR tonne placed or named unplaced
    return new, T, P


# ----------------------------------------------------------------------------- provenance and outputs
def content_sha(a):
    a = np.ascontiguousarray(a)
    return sha(f'{a.dtype.str}|{a.shape}|'.encode() + a.tobytes())


def pkg_versions():
    out = {}
    for p in ('numpy', 'pandas', 'tables', 'openpyxl', 'rasterio', 'scipy', 'xarray', 'netCDF4'):
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
    ap.add_argument('--nlum-map', required=True, help='output folder of nlum_map.py (sa2_2021_cells.npy)')
    ap.add_argument('--map-dir', required=True, help='folder holding the final 2021-base lumap.h5 and lmmap.h5 (livestock_map.py)')
    ap.add_argument('--x-mrj', required=True, help='x_mrj.npy from eligibility.py')
    ap.add_argument('--out', required=True, help='output folder (e.g. input/Nick_build/ghg_factors)')
    ap.add_argument('--cache', required=True, help='shared raw-download cache (content-addressed by sha256)')
    ap.add_argument('--offline', action='store_true', help='fail instead of downloading on a cache miss')
    ap.add_argument('--live-check', action='store_true', help='also fetch every API response and stop if any differs from its pin')
    ap.add_argument('--source', action='append', default=[], metavar='NAME=PATH',
                    help='a local copy of a pinned source (must match its pin); NAME in nir_livestock, nir_fertiliser, anga_L<loc>_S<sector>_G<gas>')
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
    zips = {k: fetch(k, a.cache, a.offline, overrides) for k in SOURCES}
    if a.live_check:
        live_check([k for k in SOURCES if k.startswith('anga_')])
    print('sources ready', f'{time.time() - t0:.0f}s', flush=True)
    inp = {f: os.path.join(a.input, f) for f in INHERITED}
    upstream = {'sa2_2021_cells.npy': os.path.join(a.nlum_map, 'sa2_2021_cells.npy'), 'lumap.h5': os.path.join(a.map_dir, 'lumap.h5'),
                'lmmap.h5': os.path.join(a.map_dir, 'lmmap.h5'), 'x_mrj.npy': a.x_mrj}
    in_sha = {f: sha_file(p) for f, p in {**inp, **upstream}.items()}

    A = anga_frame({k: v for k, v in zips.items() if k.startswith('anga_')})
    ST = np.load(inp['state_id.npy']).astype(np.int64)
    sa2 = fix_state_ties(np.load(upstream['sa2_2021_cells.npy']).astype(np.int64), ST, inp['NLUM_2010-11_mask.tif'])
    st = pd.Series(sa2 // 10**8).map(ST_NAME).fillna('NA').to_numpy().astype('U3')
    lumap = pd.read_hdf(upstream['lumap.h5']).to_numpy()
    lmmap = pd.read_hdf(upstream['lmmap.h5']).to_numpy()
    area = pd.read_hdf(inp['real_area.h5']).to_numpy().astype(np.float64)
    x = np.ascontiguousarray(np.load(upstream['x_mrj.npy']).transpose(0, 2, 1))     # (lm, lu, cell), contiguous per column
    LUS = pd.read_csv(inp['ag_landuses.csv'], header=None)[0].tolist()
    assert len(st) == len(lumap) == len(area) == x.shape[2] == NCELL and x.shape == (2, len(LUS), NCELL)

    # ---- livestock and off-land
    F = lvstk_factors(A, nir_populations(zips['nir_livestock']))
    lv = lvstk_table(pd.read_hdf(inp['agGHG_lvstk.h5']), F, st)
    off = offland(A, pd.read_hdf(inp['demand_projections.h5']), inp['agGHG_lvstk_off_land.csv'])
    # ---- crops and irrigated pasture
    S = fert_systems(A, nir_fert_n(zips['nir_fertiliser']))
    AR = nir_area_table()
    T = luto_ha_rates(S, AR, lumap, lmmap, area, np.where(st == 'ACT', 'NSW', st), LUS)
    cr10 = pd.read_hdf(inp['agGHG_crops.h5'])
    ip10 = pd.read_hdf(inp['agGHG_irrpast.h5'])
    cr, ip, FL, r219 = soil_tables(cr10, ip10, S, AR, T, st, x, lumap, lmmap, area, LUS)
    # ---- crop fields: the (lm, crop) pairs of the 2010 layout and every other pair eligible in x_mrj (dry Pears)
    pairs = sorted({tuple(c)[1:] for c in cr10.columns} | {(lm, lu) for j, lu in enumerate(LUS) for mi, lm in enumerate(('dry', 'irr'))
                                                           if not any(a in lu for a in LV) and 'Unallocated' not in lu
                                                           and x[mi, j].any()})
    CF, CT, CP = crop_fields(A, st, x, lumap, lmmap, area, LUS, pairs)
    cr = pd.concat([cr, pd.DataFrame(CF, index=cr.index)], axis=1)
    print('factors done', f'{time.time() - t0:.0f}s', flush=True)

    # ---- write (layouts of the 2010-base files)
    outs = {'agGHG_lvstk.h5': ('agGHG_lvstk', lv), 'agGHG_crops.h5': ('agGHG_crops', cr), 'agGHG_irrpast.h5': ('agGHG_irrpast', ip)}
    ref = {'agGHG_lvstk.h5': pd.read_hdf(inp['agGHG_lvstk.h5'], stop=1), 'agGHG_crops.h5': cr10, 'agGHG_irrpast.h5': ip10}
    for f, (key, df) in outs.items():
        r = ref[f]
        extra = list(CF) if f == 'agGHG_crops.h5' else []               # the crop fields, after the 2010 layout
        assert [tuple(np.atleast_1d(c)) for c in df.columns] == [tuple(np.atleast_1d(c)) for c in r.columns] + extra, f
        assert (df.dtypes.values[:r.shape[1]] == r.dtypes.values).all(), f
        assert all(t == np.float32 for t in df.dtypes.values[r.shape[1]:]), f                 # appended crop fields
        assert len(df) == NCELL, f
        df.columns.names = r.columns.names                   # column level names as in the file it replaces
        assert list(df.columns.names) == list(r.columns.names) and df.index.name == r.index.name, f
        df.to_hdf(os.path.join(a.out, f), key=key, mode='w', format='table', complevel=9, complib='zlib', index=False)
    eol = '\r\n' if b'\r\n' in open(inp['agGHG_lvstk_off_land.csv'], 'rb').read() else '\n'   # line ends of the 2010-base file
    off.to_csv(os.path.join(a.out, 'agGHG_lvstk_off_land.csv'), index=False, lineterminator=eol)
    T.to_csv(os.path.join(a.out, 'ghg_crop_rates.csv'), index=False, lineterminator='\n')
    CT.to_csv(os.path.join(a.out, 'ghg_crop_field_rates.csv'), index=False, lineterminator='\n')
    CP.to_csv(os.path.join(a.out, 'ghg_crop_field_placed.csv'), index=False, lineterminator='\n')
    FL.to_csv(os.path.join(a.out, 'interim_fill_energy_lifecycle.csv'), index=False, lineterminator='\n')
    ncs = [NC.write_like(a.out, 'agGHG_lvstk', lv, ['animal', 'field']), NC.write_like(a.out, 'agGHG_irrpast', ip)]   # jinzhu layout
    print('written', f'{time.time() - t0:.0f}s', flush=True)

    # ---- provenance (last)
    me = open(os.path.abspath(__file__), 'rb').read()
    P = [['source', k, url, pin, 'sha256 of raw file or API response', note] for k, (url, pin, note) in SOURCES.items()]
    P += [['reference', 'NIR 2024 Vol 2 Table A5.5.8.1, FY2020-21 areas (REF nir_area_2021, hand-entered)', NIR_VOL2[0], NIR_VOL2[1],
           'sha256 of the PDF the values were read from; the tool does not download it', 'pages 200-203']]
    P += [['reference', f'NIR Vol 2 typo {s} {k} {y}', NIR_VOL2[0], NIR_VOL2[1], f'printed {pr} read as {v}', f'page {pg}']
          for s, k, y, pg, pr, v in REF['nir_area_typos']]
    P += [['inherited', f, '', in_sha[f], 'inherited 2010-build input (inherited_2010)', ''] for f in INHERITED]
    P += [['upstream', f, '', in_sha[f], 'input from another 2021-base tool', upstream[f].replace('\\', '/')] for f in upstream]
    P += [['rate', f'irrigated pasture {k}', '', '', repr(float(v)), 'irrigated pasture, national'] for k, v in r219.items()]
    P += [['interim', 'interim energy and lifecycle fill', '', '', f'{int(FL.filled_cells.sum())} cells in {len(FL)} rows',
           REF['interim_label']]]
    P += [['script', 'input/Nick_build/scripts/ghg_factors.py', '', sha(me), 'sha256 of script', '']]
    P += [['package', k, '', v, 'version', ''] for k, v in sorted(pkg_versions().items())]
    P += [['python', sys.version.split()[0], '', '', 'interpreter', '']]
    for f, (key, _) in outs.items():
        p = os.path.join(a.out, f)
        back = pd.read_hdf(p)
        P.append(['output', f, '', sha_file(p), 'sha256 of file (HDF5 bytes vary between runs)', ''])
        P.append(['dataset', f'{f}:/{key} values', '', content_sha(back.to_numpy()), 'sha256 of dtype, shape and array bytes',
                  f'{back.shape[1]} columns {back.to_numpy().dtype} {back.shape}'])
        P.append(['dataset', f'{f}:/{key} index', '', content_sha(back.index.to_numpy()), 'sha256 of dtype, shape and array bytes',
                  'cell order'])
    P += [['netcdf', os.path.basename(p), '', h, 'sha256 of NetCDF content (nc_layout.content_sha)', 'checked equal to the .h5 arrays'] for p, h in ncs]
    for f in ('agGHG_lvstk_off_land.csv', 'ghg_crop_rates.csv', 'ghg_crop_field_rates.csv', 'ghg_crop_field_placed.csv',
              'interim_fill_energy_lifecycle.csv'):
        P.append(['output', f, '', sha_file(os.path.join(a.out, f)), 'sha256', REF['interim_label'] if f.startswith('interim') else ''])
    write_csv(os.path.join(a.out, 'provenance.csv'), ['kind', 'item', 'url', 'sha256', 'note', 'source'], P)
    print('done', f'{time.time() - t0:.0f}s')


# ============================================================================= API response pins
# "loc sector gas sha256" of each raw StatesOutputTimeSeries response, retrieved 2026-10-04/05. Locations:
# 3 AUSTRALIA, 6 NSW, 11 VIC, 8 QLD, 9 SA, 12 WA, 10 TAS, 7 NT, 5 ACT. Gas: 103 CO2e AR5 (Gg), 108 CH4 and 109 N2O as CO2e.
ANGA_PINS = """
3 684 103 49b551a7061e6b4c34df485e81db72f6d5ff742d8702dbc5157b088d9a775858
3 705 103 1211e278bdf1ef51e2a70b3a31b755c2e20d23c9623fb332f9baf7de2dde93a3
3 720 108 be21ac4749b467f5e0597ecf70c882f6ea63912aa307430d2004bda63a535bc1
3 737 103 ea990f2119393c3d53182878a7689b510abecb9a1c782cb9cc02933a0ac5ec7f
3 945 103 6d9cf71f438c189284cfbacbceec4ffae09adb053a78431bf0b05f557decfafb
3 1092 108 41798ef6899acaa79b00930fa77ac4ceb9f5ce9e83d08e3c8e5875fd0c471220
3 1092 109 f42d1725a2383687e4d76ee1f8a290e475cc9ceda5506de622e757f3d01f03e2
3 5250 108 fcad115b8455090972b2d5c59d130ea6f9c1bfed34eb0a33eb9e8971dfc44860
3 5250 109 56d596ce6f23ff0ac5e27f3b5326efc469ab27da0dab18736084418ffec63710
3 5251 108 a996c2d8eb928198d48c84b8713d8f09fe35cee24946e66d5e70b8e300b88a3a
3 5251 109 3f6930367add0fd678f1f116850a4adfbc1b88aef026bddc22e8a8ea63357eeb
3 5263 109 0573879ee2ac1defd1fcd85597c6708e58ee3738c7e28927068489749094554d
3 5279 109 179c9f6fb549313ac5505b331f4d9ba8c5cffd9b901eb00a6e69d4d25b4250c2
3 5300 109 a2e8d28cb101ff1781ece152e3b65cd7c9d86e927173cd00ba448b20a8eb74ff
3 5307 109 381eb687a7888f22c160cdd431db92b638435a3634d87c6d64b8eb0f653f7266
3 5321 103 1ff9443503ebeaaa667e8b9e2313b33147ce9bd51d0f6eb4ce07d9932726e6fe
3 5322 109 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5340 103 4307e6cca95d8cefcf591c09fc2f6b972a53203f6f56b7dcebc06464bdce9cdf
3 5341 109 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5348 109 33a41f3c19cd4606385bd940057ad73f48e06e07a8e0f7f1c6e2067480a76c3b
3 5377 103 9632bb8b728cb09fa9e96d8e1ac824e214db74b2db9b90b5a387df469c1bbc66
3 5378 109 b908e402f5febd76e0b9d7f4a609d62231f75c13a111ecbf9bb31ebd1292c8b5
3 5385 109 2dfb9279da2cd3c01079fcfc0f95ecb9a80eb3d17a15ebc5cc951429d35e2bfd
3 5409 103 e5a98f26913899c13bb8526f9fea1cb3e05d0977289b86fbfc06100d1f1b8b82
3 5410 109 512ad4a02175d27579410b0f058ddd34e965518b7786c7cbf83d02e9d9937d0e
3 5417 109 2a3a977bca44cd57178b262dc4c674f88e16c9ee34a893abe5984009bd5d8a70
3 5555 103 12e2d10ae10b4c1883d8047328995f7941d0c8640806691ed53aeef0c2ee000c
3 5557 103 d77c9651e0da4f9816c5122357a91c227ec8453cdd3906a53c973b8eb995056a
3 5559 109 4661c783f0ea02215155d534e7faec1a6446e42775e021fdecc45cd9831577ed
3 5562 103 753ba6d91b3a9ce943a76c34dc980e175854ae747ce55e206319883836f08963
3 5563 109 fce9699f28b1104ad2cfb4a4215564a9b4cc6335c5dd1072d8a124ac2f98850b
3 5811 103 179951afe8a4a109b17b86155baf2ec65240cdf9a559e88b244ea89472aaec2e
3 5823 103 08b86c6fe96f8072be94154fa4f06b1846134a5d9836be4e3534b7352c291032
3 5835 103 8cc24cf55f5d80da55c59d2f721bf80695684cd71509bef84cce7e563ac88972
3 5836 103 8e8cbc1051fad2fb8b3a452debe614ba5c051e96797737fe7008fd769752da8f
3 5837 103 3662d86b00b1ad6fb47154c4319d06d3427f8ba2c27f5bf06d75df7f7f14a99b
3 5838 103 57487b1dbfd0938e70e3a2c940e91e9f379e8508385a3aa555b770f497c080eb
3 5851 103 f4d7052d36bc33fcf90de8b124fc5e7d84716d6918be0b79987cfe1af39beec7
3 5852 103 95703d4a7ac707f7cec0561b99ab8e6438ed7d8d3b14708f2202da2224f66ad4
3 5855 103 c80e54b90c4136ff3440cc8cf1e3695e4a3f6c7eb2449bee26ff056d4d6a6a88
3 5856 103 40c55236275ce6984742e1658c98648f733e5cdf5e51df158cfc61b24bedb139
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
3 1238 103 fde5e846a4bdd23c56a1ba4f3da59956d9fb64f1fe46e1edf71eec2d03e786a7
3 1289 103 2ac0b326f7e3475c9e68d379eba05aac3022cfc03e1647000f979f96bcc3e13f
3 1290 103 1cc0e5013551199d036b944715eb17828ca09fbb3c00436d14f011733192f8e9
3 1294 103 2d27c577a4e96297bd82fc1cbeaa2923ee7ffbedc3ba72cd9c2f0a9fcef54c6d
3 1295 103 a45cf4dc375161fe2d2f0ea2ffbf802ef15ab8d1dc66d403f11f6c6676076c57
3 1298 103 ee3b1aa84f51dede91f162a63d2f8dacacf73aedaf48ef0d8570764add17a54b
3 1299 103 bbe4bc2c089454e8f08cb63a6bf84c0c8f63c5fe36825f578e93a189e1d74b0a
3 1300 103 64ebd36b10c53af6188edb6352df7858e7db88542130d5a92fa5512e55f5fa67
3 1301 103 b809bc98a561a2b9ac0c328526a0e9486e37f8bf46511da0e7270fd3009a223f
3 1370 103 cad3ba4f7f31a799d4866c98017d04bb618313c39bc2787e8848d4862514a675
3 1371 103 1855b5f23d2a8e8414a6a5abeae63a431b0db507602bbd22ee4d473f0c817638
3 1372 103 9093e77bf247af045e66f92c6d6e9ad8be873590e67a44d5386ce5824ecf8e27
3 1378 103 7df073155784bac50a7553b9e29ade5f7c16ad64a85b5653fea2fd3e317e7a8b
3 1379 103 b2bbcbba15b4e786bce0f83a4177e6ceda070ca76280a565d0c2b648e3ca8e68
3 1380 103 667b866890d096370265746f8e28428021435a56a60d13391d5b9b31ae77fba7
3 1381 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 1382 103 4f237b6320397a38d31521803757d3dd4f0489bdeb8b83b86334b5bd075cf489
3 5353 103 78f6156847c88671f3a0bce306c082b2fed920493f10b98273c5551eb4dab823
3 5354 103 3d6e6e52b0c74258e74edbc44514d86d8b59f9cfbd953c9ff9e6bd6a715ac557
3 5424 103 b21d4216d7bdf07c5db33c343ed82ff6b50c7cc6c08fabcb67f220f93747780b
3 5425 103 75f6305e688117368d979be5be3e6583459abcd02b949caa18ce55f2cb6bfa89
3 5426 103 9b0929f247123373da74201400db3f6099ef54c68ec207ee3c75ff665e49ce2f
3 5429 103 cb76d039e93be550d9226ab82ff759923d0fff9eba96fdaca1c59aa70dd671bb
3 5430 103 0147e561031a25ea8e8daa64bbea9ee2dfa7f23b4a6d01b8c6a7863f3beefa19
3 5432 103 51fa7133ff56164b580069a1cb6dc23522ae9a6cfbc6cd4364292a53069935fc
3 5433 103 0e594a6b74684281af588f24916be76983aeacc1398d2c5ae7a2575991ad0a61
3 5434 103 3c4bdb8e908256ab91e1df5c0ac991224b414a4d4b580dadf88a70cae43405b9
3 5435 103 1e76e772532a8584007b3221db600e94a2326c0c26acd9cb60950ef42e125f19
3 5436 103 281c38499f94f1330ed32950ab7591a65691ee2923e77f82a2b1d25c694e1253
3 5437 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5443 103 7743173aa4b00449e58c7e936bbddf66998cea82383af6277410a2f7fd0c4979
3 5444 103 d9a06f92ae618d6e8e7afd7791099fb62fd536a9072aa0239253da75f82132eb
3 5839 103 4496b7d268c26226494e8222074e7ba74bb36e35cbbe5e95fae558126c60d68c
3 5840 103 3906b4859f3ec13ad022c9f596cafbcda1159f6d79336219c76a25d493d06275
3 5841 103 569219961f782e3c9928a2e6d1dd20a41fdb801b554bee670e3440045020f61f
3 5842 103 be9b96a89a4f919392e794bc0a6fb4a211282863ccaa5eefa30f3003f8e22b66
3 5843 103 c5039f35db50ce60c15a905556cb1658da8210ff78e20caaa6b98fb632693864
3 5844 103 e0857266f378fa39b6a14b00ac43050f47e406e4909a0b4a7f5a73ea3c2748fc
3 5846 103 d6d2dd58616ea5fa3a4321b0741bca24cdd9e0ece6d9237593488993258fe1fe
3 5854 103 840926ff91f47d4dc25dfeb2d7b6fc6df6e796256f1b543302142e719a4628f4
3 5859 103 8118d61a0e25d0715c9523e91cd22b209d21ef680bed37d2518c273aabecd4e2
3 5860 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5861 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5862 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5863 103 3af00ebe617ab308873bfe9a77c572fe4f8c4259d9fb3281c7b867588285f734
3 5864 103 0852fc2507945baf8260b054ddcc4c7a0d155fd3ed88176128283fa1f8fbf567
3 5865 103 32117a2863286dbe1b59b209bcc35d2c9aa62a7f2b1b48aa3e1af3aed4655949
3 5866 103 c1289a931d50693d170699cd5ddf196161a6c602b2abbd1540fad7547c03aabe
3 5867 103 8b28a3fc488acde652a2c2267d01f78c65fb8df26899107f5e696ecf89cbead3
3 5868 103 598a9bc67446baaa2b594fa8389b45c3a0050cc94822ec2d280a3b4c1ad33d27
5 1238 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 1289 103 ec3082cb615b7a97b484d15cab459ba5560d3b1430cea9cf88a731fe4a36147d
5 1290 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 1294 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 1295 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 1298 103 7a1a4a66057a9d1b35c8e593a160eeb4bbc8b253012cb6e559fe2d1db33026f5
5 1299 103 eda7febbf54d0c8e35d09181980fbed511fc0bc0cb40f349e0108fe071c7e656
5 1300 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 1301 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 1370 103 93b4342d59cba694cf0827cfd8df9ca3277bf5367022b955bec7d3d0944cc2cb
5 1371 103 01338813e021bfcd8876ad63e89311c5c4e62a7050ab42151595a30b24b2c406
5 1372 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 1378 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 1379 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 1380 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 1381 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 1382 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5353 103 f24d476381646d16fd7e746c8e4f692ee9e3f2e95742e11ba7bbf77276f3f697
5 5354 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5424 103 7f60cb7e77c416dcda0660c2a2b0820e45969d045c9a78d961a1c5fae669036d
5 5425 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5426 103 65312176cd75c91032dfb80dc21041a6be8e4714f9ebddeeed28771813fa3af8
5 5429 103 e426ae9aff9d8b95f9dc9aba6620cdc4ffbfc0dd48cc5d6e4d4d855b457520f3
5 5430 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5432 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5433 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5434 103 14de521d159e472495d56f4e8e788dbefe12bba7691cd90092080150c795b07d
5 5435 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5436 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5437 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5443 103 f6293e103cbe798c0cc18d2fa353c483286dfed580c713f5707f135442548199
5 5444 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5839 103 6d3c9c8fc428cfa22a2f73bd3b11d926ce62dc60e71b9d99e5ac8109332e2aa9
5 5840 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5841 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5842 103 f7877e8dfccf6bcb8561f8c60cb5914d571286e7bbb32efe617c601c8a41ea45
5 5843 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5844 103 4dbbd30a955dd800916d4e4b816a0579d84d809e1cfe7f460d39ba2d2cfec926
5 5846 103 224cb102a324aa79b12ed1620be823d85d61c835a14aafa6b357b82f340e26e1
5 5854 103 b09353d591f91fe4e9600ee8d7d0aad00c2054cbbb5e87b2522564a1b63c2881
5 5859 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5860 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5861 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5862 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5863 103 f7310456132ba83faeab85ffd51a226fe138b19f91e5ccdb124a4d8130dec606
5 5864 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5865 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5866 103 58a546445a01c0273094fcda058a0c9c4f0c705dc47c7aa994090833eb9def50
5 5867 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5868 103 0d692554275d7613080627926801a548f8d514e3a1679289660b1c98e7412eeb
6 1238 103 4d2bb5e6663734b990edad8ec3f157acb904231da5339b27c0e7f83914b2850d
6 1289 103 5485276dee31f9cb3554f13c708b987e5cd3f5fe333f3e232d881fcd4ddaf403
6 1290 103 86950ab5a51368ed5737c9964b214df7ee6c83666e63b32fc41ff16a9522ebfc
6 1294 103 8127540e795ee33d2af3e4e4e4ffdaf3110a2507534773ba10f6b21e277d606d
6 1295 103 24260ea9a4b71fef70052e677c0c2261db262b82941560b67a5c3339b978f821
6 1298 103 bba311bff79314e4401080e58267b0e0dbf49af85aeca931c091486c5b11bf98
6 1299 103 80ccaad2c2536138cf14c293676a0745328e8c5eb69bbdde3c4bfa45f8922c50
6 1300 103 a59781949908262be7dff1651388111a10f3ee2998100b23e9cf1fe04a2111de
6 1301 103 92df0424ff7b3e3dacaea3b327ac7a1a428a9ee7d7a847e2e0521cdf8e85f19f
6 1370 103 f76391db1fab3ffc210aea997294f11e8073972ec987787bb2991d5d1765fa5a
6 1371 103 b23c3633d186721a40089f590ebe4bf15f36baa2aface246aa4045d5d0038ab1
6 1372 103 9220209276de04024b630b0c0431d7f16f6a17543e7b4d40e4894d1357b75243
6 1378 103 7ca7b70541662e6898532210cac1e0005490701d8608ffecf3e03c7a66dbd9d2
6 1379 103 e6dfa9be0f90461b66f0b2ea2ff81831f33ed4e5f64057bb434a2a836d3abc31
6 1380 103 3271e3cbc254a0aa1674adcb8389ceb6b6490c41e460209d2388a0776840d1fb
6 1381 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
6 1382 103 41c37c755a25c2909f58af05d909cef78514002a73768ff80621e3e6f2284c56
6 5353 103 1f2181ad4a73b6af24263c8b0156e98a37505abf0588840cc0ed34fff617c610
6 5354 103 db09148d3e18ee5c30da44e97cbef7a0f28089d85485c44f7663a64c7a0fae68
6 5424 103 62d840267d865eec77a76ca804a2d0fa6d0de79d54e86959b8e8840579c4e842
6 5425 103 8c2c215a93b5b161036eb3f2dba934ce038e09fdc69a81b91bda91123e7ab6aa
6 5426 103 08ecb1657aa4f3290339fbb1743b9adbfbb8e9c5912ad8b354de517812739aa0
6 5429 103 3136de4d9663c63752bd1e24bb7e742787b8bdcc4616929fc73217f0093f6032
6 5430 103 7c9d328c7ad028d08a720ca123ca8c28af376372f2d69cb145d7f7a515303a59
6 5432 103 d2764f5cdf541f80f9c7b5a5677abce1b3ccf923d90d96e08b079a6cf2802bb2
6 5433 103 bb18c9efd0a5c0067c23fdbff9c7c400d534458d7d809ff2fa371402e0c7d1d6
6 5434 103 4c88072060b3ec92c1f7f0b43f017ddf06b96a92485360332a935bb6acfd6c76
6 5435 103 b8bced3ea94d2630b13600aaa5d3528df168b64ef51c241c5cb455a8faa6bb0f
6 5436 103 85cf1f726a0380b79f319e50e68e4b966033176031fa3b5127928b7f15ef8227
6 5437 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
6 5443 103 2c7fe8ce98dc9f158e442f0b319e11bd2ec53ed23928c130403e3ef3f8c2558f
6 5444 103 ec6efffe587a537ff6fd18a1b882dce8642b17f0f7b4551a68a619df7a77bf75
6 5839 103 d5a53f06757836280e0d525e5ae1cad1f1bed2fbedb006cea9d2efee009c8ba0
6 5840 103 33b2c31425545e4d54801079772dfc9704ed7e3d81549c36e2e683900670148d
6 5841 103 87f3eba4699ffa764bcc381e8b2b72e1ec01e013346bbd8ed66ba2bce1337646
6 5842 103 ab866e80badbf1123e836df56014ecdf7d379048916cc52ee5b8a352281c6e98
6 5843 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
6 5844 103 786f09767ac2f747b347d44a582e00245b33dc48ad9806190dc73d43e0e0e27a
6 5846 103 6884a150f2548bea182acf7f5ef586de7fb7eebab2285bdaa1ba5fedda69efe5
6 5854 103 90d47daca4322325d2383b1db05f22227a736e532e761af686b959b51b680e3b
6 5859 103 712dd8f19cd4c302a604ed563b9ab01a20aee34518cd861a99d7b191a951be71
6 5860 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
6 5861 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
6 5862 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
6 5863 103 5e7a6873a0a1f6a5d437d4e4220d618229327232914090e007bc956281e50bbe
6 5864 103 173bb9ee2cd41f861438531f8d3abfe69af2ef1471c6cb7b61f4a39316e5cbbf
6 5865 103 b49a556e7c663a8c216b6642270dc0561f6bc9f5e480b238f77581bd13d56815
6 5866 103 2693e3a68b29b86f6c402d794c5eacc9509852bf86c645f68762953016ca2199
6 5867 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
6 5868 103 74dac7a4a62db6a5cb7140d048cb10b761cbf463d4f4b45ed84af6b7da56beeb
7 1238 103 020a289a89765c82bdc0bf70e9403c19f92824ce1d8215bfbac8d41259480adc
7 1289 103 8af9206d1c88e9de5d2a60a7b89dd9135a96613e4f653ba446182a3f569ae4f7
7 1290 103 eea03e0cb69e671d2cba64688ce379a577508e28ee730cba5ff15bb410272384
7 1294 103 9474dcb928e22bd33be17a6e70457c499cb7b32da626c4437a8ae77c0ee7506a
7 1295 103 5c248cb9a80d91615939925dfd4950fd2e6a0d085feebfb0eff5ebff8a1c2104
7 1298 103 f138e37747d33119bf8935985f1baf312c26aac97e5c201e0cfc99e19ba8fdfc
7 1299 103 629b55251d46e518805e6d5ccbde9e93960af2da4253d28b8af44ea0156b84e0
7 1300 103 69b8e8c9efc4e4f84a5f63174a69f1cb531635b0f7359d5380a395bd6c40cdab
7 1301 103 9eea28bb940c12251d30956ef8606fd083ddd8104042882a107468f8d4b357e5
7 1370 103 e3f2bd73511caa18eceeb9afc7b92132fb89143ebc0ad25dd4be95c2d54f2326
7 1371 103 b36b86759e3e229bc1b49ac9768627e11bc9597079731a05d490173e961fec5d
7 1372 103 12dcbacc80f53efe176e0d37900a3595b0d5e5868d82a31cf99dcb6b3dcc8264
7 1378 103 a78aca51ca32228749caaaec21c5d7c925201dd10435617e4658ae98fd443323
7 1379 103 edc08d141df83f8c3864e20b02eb16178ba1a3f94767e39dc60a47826048b0b0
7 1380 103 64584d041d1fca1db62f33851898d6a2982e958b8901b0720c3a028728ed820f
7 1381 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
7 1382 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
7 5353 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
7 5354 103 32da71d0f32b076e86b8e2eed2691f2f996143845f3c986ac960a1fc108fce5d
7 5424 103 98b9711e8feefcd39f42395fa5b104277dcfbd02b07a5866daf774e17a7c83a2
7 5425 103 40ecb6f5e67006b1d3bc648380ec0cb5b9586a8800ab46cfc5c1ef6cade9f4ed
7 5426 103 4af9051dcbd90506db0fbeedec7706279e53ad7ebd701abbacbc729849c65b62
7 5429 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
7 5430 103 1e3cff9854ee477549e8110939dc000c16605d7d6861a48477ae89d3b59392db
7 5432 103 dc20c0063dc5f1412fc51451a9b48c04152de89825b14be439f8358a680f8c53
7 5433 103 b82581a20689de72deabcbeb0ab56012d474ea16e5abe374a1e5f395b7bee3b6
7 5434 103 7233140b4ea23e72a64ab8b20f804c8c604a4d4a02de931ebbb368cd55c291ef
7 5435 103 0cef981e7b29772b371364fbf3620934ea48db9a62805a272a6a4afb1488131a
7 5436 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
7 5437 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
7 5443 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
7 5444 103 a95dd4a10bb6cc70f9d5b555f09e52be3a90fe5e668f89658183aa57c6135c78
7 5839 103 df2a9235d7197931e14efbb06269d7df3367e6982dbff2f92eae6beb4539adf3
7 5840 103 b006725cb3e74fdd7470a0b8e0503533166fe0630f6833715d89e42cd39cb5fc
7 5841 103 07c84b651cf16ce07825539e1e7b2cf7c8a4dbd32654b781942757c55c00de5b
7 5842 103 3a0c8c1c0711d8a007e2fe993ecdeb1ce2c5730d93022c2304fb9692fea977f5
7 5843 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
7 5844 103 131ceee26253f7a4e01298a8d57ef14034e4942b7b846acac76e78048c52a7dc
7 5846 103 43daf2b6f093f16235ac5c88daadcda33ce58dc751ed4372a9399f1310a0027a
7 5854 103 c38cbd1fde9997c3fea8159793f90ec7b2add7b932e1b834fbff16fd0ce662b9
7 5859 103 95da59bc284417c8d8ca31f4721480fd5a59a84cbd53d845ffee343daa6a0195
7 5860 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
7 5861 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
7 5862 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
7 5863 103 1e39d713d04b735e6a0b86a39f0eba56f72ffa73e62dcfadc3cac9559599cf26
7 5864 103 df46a703dbd8763f8c7608e9f28636bd9716b82367c7ebe439bfa949e0ef7a97
7 5865 103 024641130a686cc0b2123923b68918d8c0fcfc61cccff0e9c0d5612a1d121c7c
7 5866 103 e0da7a5773c6f8bfe3aa4aaefc84ab9f0e288460f3e9e790d7859857a569208e
7 5867 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
7 5868 103 70979b202885d63914ef5ef863a4a7f31365deb2ce55fcd0c55b7392a519e0c3
8 1238 103 127c3a5039ed320a2462ec00183546ee4d0b2d27fff13ccfbdd2631e2805c949
8 1289 103 85832ebed95315b2260c51295a7969d354f99b0a8b34dcfc67eaf9d2b40185da
8 1290 103 b2f841d1f98aae92e164a9c5bf040f262734329cd0a28dda702e14c22eb26531
8 1294 103 55d1b5f9dc76e3bc1b29ed0abe4168b2f1ccb8ce4c071d659cfb569d4cb4a343
8 1295 103 ef2a04e48577c54ec0fc9f00314852a360317050fdf243202eb3dbc0b4c18f1c
8 1298 103 80bda7d7df3defcee902caa3fdf478292afabcaec009f938e366369dd524971d
8 1299 103 76829d00e2507ac481531cc7b2519aff5a2a0083a015453f52195c53265498d4
8 1300 103 2d4bf2ced9ce9f6f8ddbdbc185383a59335afc7443f82afb5854de31a151d8cf
8 1301 103 0be49c51da4054415da38725d3a70d545ac63ec554232cb3d0fc6154f8772a40
8 1370 103 28d87d164a94e3aafa9898ef1c963ddc108957b66ce419f69776608351b83298
8 1371 103 88be14e9c7f92ee546a1dff6e27131b62dde3f86f89c2c610d033ee86997fc34
8 1372 103 5dc082b799a9d14a2397d9c81eb19b25cc57e59e8899d94b2fff5b6d24bd2366
8 1378 103 dd5aeb996ebe5d13d3cdb801af7f45b54533e6a6878cb6bcc126093c0e7e6f1e
8 1379 103 4248f803d223c80567b7c9668c56fb6ec37e377970096d83f947b67b83d39f80
8 1380 103 20498238c402f4201c70d5d58903b78092cb91906954699ca0345fa6b2ac372e
8 1381 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
8 1382 103 f4a80164cd932c147c6e4a05d58cbf7a1f747f4957de213e3d08fcdcc8dd1019
8 5353 103 fa23cf7fcaa090f3359fad2d13becffbdf0da1f0b60df0f2cf75108751ca948f
8 5354 103 549f5395e5fb67a2e376bd02c3174fb1fa6fac9b852d31987a3b1680e8c0f302
8 5424 103 1ada606ec41fae70418b61969724f6d909161efa8323c17e6892a6cbb2405668
8 5425 103 60fe2653e522306465cfb324950c73b477624792c8507c0c46910593646bc136
8 5426 103 470c3d07c0132fe29baaf518cf572244a7ae1d6da5616aab6d48ae0a982ef87e
8 5429 103 6dfb356164e6d7e4209c3c8aaa33032693a685741d7243a5506f79a4e22e5abf
8 5430 103 7ef5ac7186a8b4506c5e76e58be33110dfe93d421cf47cf6d9204929a9f4d8fb
8 5432 103 a30a0791c11aed02f19f3b29cf3e5c6b8eba97d96073999dcaaae7048cd4fcba
8 5433 103 6ce91c325ce21a58cec455b790082b0b2a8c84da7ffc7c583eca0c0477aabe2c
8 5434 103 288c46135830d6a907f89839a9469510a058220c430baa9b1ff2e1ea9f63a98a
8 5435 103 c738365bd9c668e7463038c3bb525a1dbf077468677873119fb7777b84d75081
8 5436 103 af1f9cce35b8d4d459188561cef6e2bd18494feb860819fad18598925383b11a
8 5437 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
8 5443 103 8d446145c827ce9706c7b866683d26fc1c9f5867d7142e869c0511539dc35d2d
8 5444 103 ee04de447f10c7e6045588504322793bd5d9110f943fe1883b7c03b872f74d74
8 5839 103 ccffa5730b04e123d687530a35e94ef9e5c77c11373ff0569fc986d4fe61fd10
8 5840 103 964eca502af790d74378da71030c8a999ba0f56d0c6447ce66fec00990c33144
8 5841 103 f36d1f404339ddfe8e74db1bb7686247aef9788e9d6fe9a688c1366d5c75b900
8 5842 103 bc1b7e32dce4b6ca3651746db9f196c3ccda77f4a9375ee809dcea4ffd41d9e0
8 5843 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
8 5844 103 77cc85301cf87a4cc2555073ba0d32bfdc6bca49fe9e6a43b8855bcfd2d26c61
8 5846 103 2d53245df84c3840cd57dc3384ebefb4b9196c1b699d48ee2a83822a9140591d
8 5854 103 605f463dadc4036f64831d4183a7d43dbc4a82d8f29acc7a06700af10f162db6
8 5859 103 7bc71eaf0c28e81eee72189b8f8b59b7abb83b7a994c9e9a7bbc5499a4568d4c
8 5860 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
8 5861 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
8 5862 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
8 5863 103 9996f6dd02ce33254dc3cd00c297adf55723471e804ea1b6a9c58936f06d19e1
8 5864 103 4ac99ed3a8a194bdd4558585c879cbc46e70fb402cc2efce59e892d3a5756a44
8 5865 103 f5d75fa98e2a43fe4ff02e39d5bc77385d9387c5ee29f9f3b42f874e624f8efb
8 5866 103 35dfed4edba72eb1f194b151741035f0b79744702bfcdaac92761493f8c5f5a1
8 5867 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
8 5868 103 52aca7f094f7705115982c870270ed16aeb491e324df5bf964ad4c073005abbd
9 1238 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 1289 103 14fcb1dc12022ee59059fa9fe6292ecfdf42ccfc00e098ce23c99fe8005b288a
9 1290 103 326536bb72568900d316d33556cfbab8dabd33db71060e1d0e43e94407af18ad
9 1294 103 d55f50b887af75c0f57186f3c6dfbce603d7742ae76af6e12b29bb9b5265083b
9 1295 103 ee040960fa44150327102d0bd215681691a35bfa90ccb15004c83f05ec7bedf3
9 1298 103 b6e466d0f551c0bdf56ad155e6624fcce38ef27071dc5bc961477b291a4f9a1d
9 1299 103 3d0cb7f6c9c74c39df62d632adf4a2e8249836c1061d219e4ae084fe8d09946d
9 1300 103 80a9df0971ae26ec750df28af815792846bf9355fb79378178c1160f799d6770
9 1301 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 1370 103 e4b0047b461f351a7e0117c21270529ae1464d9e53a63f13ae67c45bab028b91
9 1371 103 bf0d412ad6c2b2aa0ea09df0d5586b897b1a3355d6db2002763e67aa6a9c422e
9 1372 103 7c77bbfc3227de6419648fac4160aeb29bef865439166849981894ce21f1d47a
9 1378 103 73d11ea2f4494197af9b3f632dedbb6054c2e03067c688bf15eabef30307f0cf
9 1379 103 263b0dc2de852d904ee8f3115fa10a4c9beba68ce99df0bd979bad4a70bb18c2
9 1380 103 53e0b3e7681bdf6045fc368d1563bd3d3fb2ffaeba973684102b0b5388797385
9 1381 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 1382 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 5353 103 13bc04fb1d1297557a62bc27d65eb13c4451b5e03144cec0527cc792d492a111
9 5354 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 5424 103 e2c152526f521d2dda00d42ce9dd20ab2e0391dff376a64b7d71d880c5a9d4bc
9 5425 103 b0bc237834f5b49f3f90019b0a5bfd903e29ce0a950ff4d252a8a3420d075f89
9 5426 103 c9197d60c0693c3f40d6afc855dd8483b9663497ad5f0773a104f958e95223ec
9 5429 103 2af30825bc5ca9ad63014062a608c0b5f0051cbb0d0c685af66c9112f60f7eea
9 5430 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 5432 103 34c17bc4d364340a1187cce9623e83c3ca66a1b3671d26f6f618e771d74db96c
9 5433 103 87d1f87f849856570a2cddcc24eb8dbecee90af455f9a6fda62226d27dd31ac9
9 5434 103 c2b58d19aea09d16c4788705d1022c2116b63478ab9ae34f3f897562dde4c039
9 5435 103 9a030b5d9f7f727fd6725ad6b8dc3bb226d38469044a2fb9b13f0790303f34ca
9 5436 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 5437 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 5443 103 f001f2fcd8afd11b85c4532c4ef7c5c3b5a7ce07ebead6dcb78cb924b2a3b223
9 5444 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 5839 103 e1d8c51331463a75125f058db5838db11d6888006464c714a7eb2d9f1f0a3e4e
9 5840 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 5841 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 5842 103 f079f6f522cd2d318de70e5e8f7f8b7362f50399558225e194678cba53e13a19
9 5843 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 5844 103 2656b2815469c65c754ff03eb1e2d991b85d86693b1bfc1bed851fa51bf065c9
9 5846 103 7661937697fc5d89e064864efb7165f3b8cd34750070c0eedada89084cd61c76
9 5854 103 0ef38e5ea9622501ad13e1fd7292d808904788b046335299d5e99db51df2b604
9 5859 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 5860 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 5861 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 5862 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 5863 103 00efb6966d39e00fa0c6f4b79f55a692c49229a711dc17eb3ef6c9965ad86734
9 5864 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 5865 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 5866 103 bfc0feaf0be1ed011b501f46100099806e3026b7166d7aab8a43cd4143cfc31c
9 5867 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 5868 103 8a1387cc88977dc7378402ac9ad382dbf2c3ee01734deac064348526d0ed2c42
10 1238 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
10 1289 103 51d8d50ecc541462cd5c222984cc26bba1de9f229fd88e92b2b47fea6057d149
10 1290 103 3c63d006faf53b9f1288f21991307916a543a7b2ef157e9ce7c4628fd69be0ce
10 1294 103 2d5316dda517ca61ca21ff48898e87b3c7c635d2429edbf0e979363708186055
10 1295 103 8e273df06a23d96e28f21ce8155642e241d286efa20ac06a48262832ba41a093
10 1298 103 2821213144b7d4ad1c3843c5027b562cfe2c5b4178a8deab3e83e72e0389038d
10 1299 103 f26d85e3be9fd02077b80ecd09f65a96f6915624f785f00f637f30a273b6178f
10 1300 103 3afc654230bc8c6abdfdefa52fbcc9829ed1fba624a801955c12c0e202d0cdea
10 1301 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
10 1370 103 81d79d1b7e752f14616b7c90d4fec07c71c0d305e09d65a4a0814fd00df8e319
10 1371 103 ae428fb5471f9bd8fb50a32faf7d799d139c06a6a0a9db1d9ad54d03495a2a88
10 1372 103 2732a538a3629574415ac794f8601b26de3004c5c20b391054a41931a2de6d4b
10 1378 103 d7887c904aea6de47b0efa2a54d233c9e9b89da828e454dc99303423a4d0fcf2
10 1379 103 53f62347cae61e64084c0c68d35f97ae93f85ca2ba9b40b218abc63105dd2508
10 1380 103 bdc2f68ec5cf7f5236597ab88dc8f7bf41564e7ec82924ca8395aacc7de51d16
10 1381 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
10 1382 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
10 5353 103 4c95ce8f2bb7f0a2d1d7b83050a5a4835dd9d0286bb5d0aa0003fc38e9ddc377
10 5354 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
10 5424 103 33f4df195ddb3ffffa0319caa85b4248f085dd1e7e1ab40bbfb4eb84c4d2d4b0
10 5425 103 ddc89e56bf135c5743c1dd04dc40343ffd39f68baafe611c23ef43fa1e718ee9
10 5426 103 c50b27bec429853c1dede3576111134805e5f12d14ad2746bf87a9a7a9413c6f
10 5429 103 e28031393a359e0843be8cdf7320c3768eebaa91605c75f534d3d13fe6b18e41
10 5430 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
10 5432 103 889cc8b21c1ac2cc055272aa6c62e4080014855683620b688d872ee56adb867b
10 5433 103 cced61c9d065dbba6bef389d7ba45ce23f92d33c1c3894189637e074b708dfda
10 5434 103 c098627e0e0cb53a3cae7164ae3ff8fd557fb050c1052fdfa84af910ecd83b51
10 5435 103 bede9475dcbc17be3d73630be3297b09083ca7add16561ccfaa19b99c7f654ba
10 5436 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
10 5437 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
10 5443 103 4054afb65a127b7410ca0510837137c152e86e56f7075eb6d02bfd66a0ca0422
10 5444 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
10 5839 103 e0399ba328a8cac66d212441c095b8102f00b820b55e58cb03dc205ba9a5e10c
10 5840 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
10 5841 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
10 5842 103 32b3ff7df5fefa7cd3a3bb1a52d42ab2828152745d8d4b4b059a1f8ff6417b88
10 5843 103 382776c63fed8d178276c3957d33921af95cf073cfc265dc920c19b5f0121e49
10 5844 103 df870e394f065c56912a76a03cef3e92d0bb8ac8de11e2fc6ad0f62a2ab367ec
10 5846 103 b581ecdf36811d51f1978aaf16a9f610e848096ad167178c10c301c13f57ca5d
10 5854 103 b6f2f6de84f2fcd3a7be934cd7386ba607ff02ac7b9d9247710e2c6b8f895975
10 5859 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
10 5860 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
10 5861 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
10 5862 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
10 5863 103 ce7feff54b91e402b6a4b0e2dc854a299c1b771583056ad9e5c0eebe5cd55736
10 5864 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
10 5865 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
10 5866 103 ec5180e3c9762c7f303db98b4ccc3274afe1dd6fe6c51094442a78c705219653
10 5867 103 6650ddb768d306994fd2deef2734902c7d754fe8b7aa263a5fcfb2cf6881f612
10 5868 103 ebc7c5c20d57982b09816c6fd8761b42d33b4cdc8c80ad2cb8c80299fd2c13f3
11 1238 103 b8ad2b3116d585da3b6c791ee1b76b52c7bdf57817e0927fe61a5ec1514e1775
11 1289 103 5df1b07312d2b3f7fa66a784e7792c743fc6a11743ae54b8c69b88faae805c19
11 1290 103 d5c425fd588c46bd57b56b870294705f53e262edb5aa1350f0d8d7f4a4fee4e5
11 1294 103 d59520583c7e0f6fba9d7467e9e0150bb439903d416dd41033d2cbeef9650414
11 1295 103 be4648bd1071b3b8eaea883e159bf7b217c8d48031aca26476f4268c89774bd7
11 1298 103 f2e93499d8402867ace6b2348d3515b076f1d1eccb7ca36aebcc6765c4ed01f5
11 1299 103 34d69ac7154534dba286feb328210f4c72a75be7aed6659103a8a4585ff7b62b
11 1300 103 75dc6d10899e571e194ea596d2e07b30f511dcbcf9783bcb481ef831619fc69e
11 1301 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
11 1370 103 ed1a426aa225b7ca4386b6e2b440ba4191a9b698bbbbcb82ecba0c1d5529b892
11 1371 103 3532b1d99548e49cccb0943bad6e5e1becc3251290ab8d7d9f6ce0076734bcd3
11 1372 103 70c44c2debf39baa4d6964394cc5001d2bdf09bf25060bf8690f0631da936ce3
11 1378 103 4df1bf4e12fc986d1774867323b6507889b7e8e3adc2350423e557d002ee4817
11 1379 103 56ce636995114d7280f370ac9bcf619b45d696dae6a327bedf1e15c309650277
11 1380 103 7919d4eb65241a2c1e905a63ecd211b00f81288726831281ce48cdbaec33d900
11 1381 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
11 1382 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
11 5353 103 6ff1151e672641e9e13accca38997e99102edaf1d8c4bc2d602a4b3404f8ee77
11 5354 103 2fe549f661edcbe90097f2b40bce8089733860862cea4f020e20ed7ec448373d
11 5424 103 e0b336f0458b3142fc99b7ea3127e4916a8367745cc73608ce80a98aa89bb3c0
11 5425 103 a56e8e3b838785b90fa831d2adef158646db13263c25fd0a25071266f65039c2
11 5426 103 419b9bb74b71730ad32797979900df376db0ffbff7e7735dafb20bef7a0b2bf7
11 5429 103 105781d248fb7048bec81b92589dab1fe119f24e26384ca6754268417722fafd
11 5430 103 c5ecbaf597b29617edc1502e292fcba4b1ff288e3d601439a6b65d979184cf63
11 5432 103 d0de83446b0521fb54825384dc2bcea7e4d3c21175ea187352de9778959a006e
11 5433 103 30b67f58e25aaf910079df1a8b16a182977c0d2c5adc08319894bc4e5ecb405b
11 5434 103 b38d133e8a2db48724c650618245ee129b6f1b7cf2839565e017be8629722071
11 5435 103 c92424ef244f62fdff8889313d50e145853170461e7dfe6b4a8eddb5b831485d
11 5436 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
11 5437 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
11 5443 103 1f3f38f91d7784316372f9094ec796dca95e13ebd9976c65c5b0fbc912b3e842
11 5444 103 7f25c6b7e5b3b95512530ec4bb9bf4d7ebe36f10726117b31729ac57e30f405c
11 5839 103 ef62e6fc67b6588221cbefcb9ebc96116e0a9852ba2ada8cc3d0aeed79da21c9
11 5840 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
11 5841 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
11 5842 103 b553aa4a78cc385a6a5a8162e40f057bb4c5529418b779f415d455b37f06b683
11 5843 103 09afa3cb3c892d257334e726e43b4648ff2ec7a01734f1b73a982054f47195b7
11 5844 103 1e819d97341dd718d89a843fc17997a2b2ef91b8607236507eb535fff9dfe2e4
11 5846 103 02cf10b07a11442e5b07efd2635138c7bf1f721ba07e304c6bf3b292d02717a0
11 5854 103 8d217a1402bd0439500c6ec483a46456f414107b90d7b77de9ab52c469581549
11 5859 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
11 5860 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
11 5861 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
11 5862 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
11 5863 103 ed46ef1319e83bed748670276192ad00e2a34222a1d02585a316f7e020877126
11 5864 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
11 5865 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
11 5866 103 f7897c22c8a8a257acd3027e8bfd560c4529d6b29bc0f53596b212f4aed13a9f
11 5867 103 a673c2955ef67f79d03a4db5a17371f99781fd30845ae2a0a0b06c86341d4c08
11 5868 103 44853369e7bb7481886acbbcba568cd00b9ed061e6fa75e2840c1039f981c2e6
12 1238 103 9f4d3c8a2b921e184fc52f87c22e5b2c9780424085ba6fd1173a83a783175eba
12 1289 103 93d985b2ec4b215cff1b35609ca483bf88f04bef32a857d489d8837186c46bfa
12 1290 103 bd04dff3d84282a71df02f840b1b8d2cf07f3647051c7b84a89c55a2a6fc00c7
12 1294 103 5c73a1a9f209c85c3a63c3196a833a1232cd7279fcd1d53e718d9ca09845f533
12 1295 103 593ee989977e72159088d6953029c2dc2d8210944e3b6aee8ae4d94dde009ef0
12 1298 103 9700c7d7f26f8e99d03e37383f048e0ffaff49eefeccf0417bec0aaf82fc04e0
12 1299 103 5da696078dd47d47761fce6b6423fc2c078a7f180a50b5331f9d57ed49830024
12 1300 103 bb369e7a65edbb99bb71220d342238528bd8dd7ac14f3d1685162a5c68d47be3
12 1301 103 f01a5be14d2329e5204a310acc8b7f401258e85002134de637536d756564d45f
12 1370 103 d820797659a957d91fc14ecac96301994d5c0b89491328464be83a06a8b9461d
12 1371 103 e16ba04815fea82fbc2eb70434e294fdb1f9542e494828cded3dde665514b8db
12 1372 103 857f83b30a7aee1d3bce72c115a48727d4ac4f9e30adb66009a1836638b67c34
12 1378 103 a9deea6bacd63a79f9cae8ec2a14e8db742fde05e45238ebf7a45b976a7d658c
12 1379 103 813db1f3cf2417eb444c449617f09e79c4bb736c079c73e3b71a46b6505ac21d
12 1380 103 1053efe1dc59fd3eaf18baa27e0e9d77ed1c5e28e1179a6a1b9396b841b1c4db
12 1381 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
12 1382 103 903b0809670288dff8735885d0b3b723723e3719103b3eb33028097ba3a10b83
12 5353 103 f5a3e3e7c1c4df185928c101adf84b8d93ae6acfd9f9a8d1acc9ef298d13335a
12 5354 103 a30c6ba9b343a7445118a189384aaa9c8362f4e742778ea058e26033ad65ade9
12 5424 103 b6047d61f242ab61f2e18fde8a33bbc05ef234e14e7be91366f9e25ad4a4e29c
12 5425 103 56f315da0e1434901fbec64ec808f5da3d75f7137ae0958024e057086e85ab6a
12 5426 103 bce9b501911cc72cd74279a9f74657bd1931694ce0c189881f00a0e14b3002dc
12 5429 103 07f895f45b27982b3a8feaf1a0b749e275ff74fe9d213f662d0a0dd127589047
12 5430 103 9845bb7523b60e31f163df8bea281207eaf3aead065f1d4df6e61cf4983cf0df
12 5432 103 0ec70b9ea81def0ec62be2c0468b9ea5ffcdc01dec319eb961a0238ed6093c1a
12 5433 103 2a8b81a9f497337a216728c91564004881d783e7e8924fc63e2ec492cbd7419a
12 5434 103 d70a79d3c9ef8ddf497400cd56426f25fc27a2a61899e5f95df63b70c9d6320a
12 5435 103 391e619ffb9585690b2cc83703575a4b322420625e74c0110235696118650c6b
12 5436 103 e1944cd36b306500f913a343433eab52ad99bbc5d9a34faeae9ed4c263a66104
12 5437 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
12 5443 103 f710fc42c5023b771ddf2574d08a2f372efee8b2c5b08c566a3c330f9d1db07f
12 5444 103 85e48fa1e04bc0dc285f15be173633c0f42c966897f2d4e29818c52a3ba14c68
12 5839 103 cbdd4d94976e0e047bca9855cf18d2c5952b03c2f5efff527d4d40569f6bbcf1
12 5840 103 395b9b23eb6e7f1d72cc8ef8e5413c4c47aad9cec2fc1735ed77fda1caa31d65
12 5841 103 a88ea0d00b747849b145db8724dd63d7c00e62702fc108663f3ad2b8c39093f8
12 5842 103 ef62f90d8b09b2f373a792a1b4c8b4ed376021ca487d9ff9e939f5dc9b7a65cd
12 5843 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
12 5844 103 f093216db1ad8aa460ccf9aef6d88bb3ec24ad9d9c569d6936b120c3e32cd0bc
12 5846 103 dca52421ac3b0d41b94a76004ef1cab6803a7d8289d754b768fd311f0692c318
12 5854 103 1406582b25fb914a67984dc11aab98fe24f75bc6588336a3d0c673805dc0a211
12 5859 103 29ceb5fd8df11054c90169566637a64a60c12fea641d74faeb8226c9c41e1338
12 5860 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
12 5861 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
12 5862 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
12 5863 103 3aa579523e6814dc57bc984e6287efcd2bd6e9ae2a0fb716ec9092c4dcb51a9e
12 5864 103 9b904286ecd48aeb277aa512f87d0431c4bb92d535cc0465d8b3259aa2954ef5
12 5865 103 f81c97802c99288e8730a26d06ffb51f1e624893ede5828ab829749a1090fc54
12 5866 103 57d8c3c4f4831f8947db89bb4d63fa45567b4d00538bc177df4446df6133e76f
12 5867 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
12 5868 103 1049e4ef92b9ea8ff88398815d3354db287ff8e51784c841a4c3f81f4d71d153
"""
for _ln in ANGA_PINS.split('\n'):
    if _ln.strip():
        _l, _s, _g, _p = _ln.split()
        SOURCES[f'anga_L{_l}_S{_s}_G{_g}'] = (ANGA.format(s=_s, l=_l, g=_g), _p, f'National Greenhouse Accounts API (NIR 2024), '
                                              f'{LOC[int(_l)]}, sector {_s}, gas {_g}; raw response pinned (changing source)')

if __name__ == '__main__':
    main()
