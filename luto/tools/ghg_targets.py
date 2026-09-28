#!/usr/bin/env python
"""Build the LUTO2 split GHG target series from public sources.

Sources (public, downloaded at run time, sha256-checked against pinned values):
  - Australia's National Greenhouse Accounts (ANGA) API, NIR 2024 (full CRT tree, sectors 3 and 4, AR5 GWP100).
  - DCCEEW "Australia's emissions projections 2025" chart-data workbook (Baseline scenario).

Run as a plain script from the repo root (do NOT use `python -m luto.tools...`: that imports
luto/tools/__init__.py, which imports luto.settings and the model stack):

    python luto/tools/ghg_targets.py --out input/ghg_targets --scope-matrix input/ghg_targets/scope_matrix.csv [--cache DIR]

Dependencies: Python stdlib, pandas, openpyxl (both in requirements.yml). No luto imports.
Output is deterministic: sorted rows, fixed float formatting, LF line endings, no run timestamps.
"""
import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import openpyxl
import pandas as pd

# ----------------------------------------------------------------------------- sources and pins
ANGA_BASE = 'https://greenhouseaccounts.climatechange.gov.au/api/'
ANGA_DEFAULTS_URL = ANGA_BASE + 'Defaults'
DCCEEW_XLSX_URL = ('https://www.dcceew.gov.au/sites/default/files/documents/'
                   'australias-emissions-projections-2025-chart-data.xlsx')
PIN = {
    'anga_defaults_raw': 'dfb118e49a3676b2e6831414b1b763bae2631383784313f6d31075ea5ec9c430',
    'dcceew_xlsx_raw': '594c37976cedf087f5dbbcbe53a73f704fb602abd45a615e8f9204242fbe25f1',
    # sha256 of the canonical text of the downloaded values: sorted lines "node_id,year,gas,repr(value)"
    'anga_all_gas_canonical': '28b37ae7d3551c2f2f4bb9d4e8662b9d5fcd048154eb2988b330a50d3e130984',
    'anga_by_gas_canonical': '79d7e043a9342cd567a935b83419a190e6c33539d97f34b79216bf461ce5bac9',
}
ANGA_CHUNK = 16            # API accepts at most 16 sectors per request
ANGA_LOCATION, ANGA_FUEL, ANGA_DATATYPE = 3, 1, 13   # Australia, all fuels, emissions net
GAS_IDS = {'ALL': 103, 'CO2': 107, 'CH4': 108, 'N2O': 109}
SOURCE_DATES = {  # from source metadata, never from the run clock
    'NIR2024': '2026-04-15 (NIR 2024 publication date, STATED on the DCCEEW NIR 2024 page)',
}

# ----------------------------------------------------------------------------- design constants
YR_HIST = list(range(2010, 2025))
YR_PROJ = list(range(2025, 2041))
YR_HELD = list(range(2041, 2051))
BASE_YEAR = 2021           # Nick: FY-end labels, base 2021 (LUTO year Y = FY ending Y)
ANCHOR = 2024              # last NIR year, splice anchor
CONVERGE_TO = 2030         # LULUCF hybrid convergence year
BUDGET_WINDOW = (2021, 2024)
FC = 'Forest conversion to agriculture and other land'
AO = 'Agricultural and other land'
FO = 'Forests'
LV_T1 = ('Dairy', 'Beef Cattle - Pasture', 'Sheep')
UNC_AG = [('3.A', 'CH4', 0.246), ('3.B', 'CH4', 0.374), ('3.B', 'N2O', 0.548), ('3.C', 'CH4', 0.112),
          ('3.D', 'N2O', 0.559), ('3.F', 'CH4', 0.381), ('3.F', 'N2O', 0.381), ('3.G', 'CO2', 0.539),
          ('3.H', 'CO2', 0.510)]   # NIR 2024 Vol 2 Annex II Table A2.3, printed p.19
UNC_FOREST_CONVERSION = 0.279      # NIR 2024 Vol 2 Annex II Table A2.4, printed p.21 (B.2, C.2)
PLANTING_LEAVES = ('4.A.2.2.i.a Hardwood Plantations', '4.A.2.2.i.b Softwood Plantations',
                   '4.A.2.2.i.c Environmental Plantings')

NICK_DECISIONS = [
    '"Source of truth for calibration is the national inventory (NIR 2024) for historical years and the DCCEEW emissions projections, baseline scenario, for projection years. LUTO2 is not to depend on AusTIMES CNS25 outputs for calibration. The existing `GHG_targets.xlsx` series is retired."',
    '"LUTO2 will carry two constraints: agriculture-sector annual emissions, and net LULUCF. Each is solved for separately."',
    '"In historical solve years the two series are benchmarks with a stated tolerance; agriculture emissions are meant to match by construction through activity and factors. The constraints bind from the first projection year."',
    '"Agriculture is matched annually. Modelled LULUCF is matched as a cumulative budget over 2021-2024 in historical years and annually in projection years."',
    '"All series on AR5 GWP100, matching NIR 2024."',
    '"Data is reproduced from public sources, not requested from N: or Brett Bryan."',
    'Base year: "FY-end labels, base 2021"',
    'Extension: "Hold flat at 2040"',
    'Splice: "Ag offset, LULUCF hybrid"',
    'Plantings: "Leave all of DCCEEW\'s plantings in the exogenous series, and count LUTO\'s term (3) only above a baseline planting rate. That treats LUTO plantings as additional to what\'s already assumed in the baseline, which is the natural scenario framing: DCCEEW\'s baseline is "current policies", and LUTO\'s plantings are what the optimisation adds on top. The baseline rate can come from NIR history, for example the FY2021-24 average rate of new plantings, carried forward as a disclosed assumption."',
    'Unmodelled agriculture: series_agriculture_exogenous on the agriculture row\'s left-hand side; projection "Proportional share"',
    'Decision B: "Direct clearing moves from `LULUCF_MOD` to `LULUCF_EXO` in the targets tool. Term (4) stays on the LULUCF row, so any clearing LUTO chooses is additional to baseline clearing, on the same logic as plantings." MOD: "Yes, MOD = 0"',
]


def fy(y):
    return f'{y - 1}-{str(y)[2:]}'


def fmt(v, nd=6):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    s = f'{v:.{nd}f}'
    return s[1:] if s.startswith('-') and set(s[1:]) <= set('0.') else s


def sha(b):
    return hashlib.sha256(b).hexdigest()


# ----------------------------------------------------------------------------- download
def http_get(url, accept='application/json'):
    last = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={'Accept': accept, 'User-Agent': 'luto-ghg-targets/1.0'})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # network retry
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f'download failed: {url}: {last}')


def check(name, digest):
    if PIN[name].startswith('__PIN'):
        print(f'WARNING: {name} not pinned; observed {digest}', file=sys.stderr)
        return
    if digest != PIN[name]:
        raise SystemExit(f'sha256 mismatch for {name}: expected {PIN[name]}, got {digest}')


# ----------------------------------------------------------------------------- cache
CACHE_DIR = None
OFFLINE = False
CACHE_INDEX_NAME = 'index.txt'


def cache_file_name(url):
    return sha(url.encode()) + '.bin'


def cache_update_index(cache_dir, fname, url):
    index_path = os.path.join(cache_dir, CACHE_INDEX_NAME)
    entries = {}
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line:
                    continue
                k, _, v = line.partition('\t')
                entries[k] = v
    entries[fname] = url
    with open(index_path, 'w', encoding='utf-8', newline='') as f:
        for k in sorted(entries):
            f.write(f'{k}\t{entries[k]}\n')


def fetch(url, accept='application/json'):
    if CACHE_DIR:
        fname = cache_file_name(url)
        fpath = os.path.join(CACHE_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                return f.read()
        if OFFLINE:
            raise SystemExit(f'--offline: no cached copy for {url}')
        raw = http_get(url, accept=accept)
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(fpath, 'wb') as f:
            f.write(raw)
        cache_update_index(CACHE_DIR, fname, url)
        return raw
    if OFFLINE:
        raise SystemExit(f'--offline: no --cache DIR configured, cannot fetch {url}')
    return http_get(url, accept=accept)


def anga_tree():
    raw = fetch(ANGA_DEFAULTS_URL)
    check('anga_defaults_raw', sha(raw))
    d = json.loads(raw)
    root = d['sectortreeParis']['Nodes'][0]
    assert root['name'] == 'Total UNFCCC', root['name']
    nodes = {}

    def walk(n, parent):
        nodes[int(n['id'])] = {'name': n['name'], 'parent': parent}
        for c in n.get('children', []):
            walk(c, int(n['id']))
    ag = next(c for c in root['children'] if c['name'] == '3 Agriculture')
    lu = next(c for c in root['children'] if c['name'].startswith('4 Land Use'))
    walk(ag, None)
    walk(lu, None)
    return nodes, int(ag['id']), int(lu['id']), sha(raw)


def anga_values(ids, gases):
    out = []
    ids = sorted(ids)
    for g in gases:
        for i in range(0, len(ids), ANGA_CHUNK):
            chunk = ids[i:i + ANGA_CHUNK]
            q = urllib.parse.urlencode({'urlencodedsectorlist': ','.join(map(str, chunk)), 'locationid': ANGA_LOCATION,
                                        'fuelid': ANGA_FUEL, 'gasid': GAS_IDS[g], 'datatypeid': ANGA_DATATYPE})
            p = json.loads(fetch(ANGA_BASE + 'OutputTimeSeries?' + q))
            if p.get('error'):
                raise RuntimeError(f'ANGA error {p["error"]} for {chunk}')
            out += [(int(r['s']), int(r['date']), g, float(r['v'])) for r in p['data']]
            time.sleep(0.15)
    out = sorted(set(out))
    canon = '\n'.join(f'{s},{y},{g},{v!r}' for s, y, g, v in out).encode()
    return out, sha(canon)


def dcceew_tables():
    raw = fetch(DCCEEW_XLSX_URL, accept='*/*')
    check('dcceew_xlsx_raw', sha(raw))
    wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
    modified = wb.properties.modified.isoformat() if wb.properties.modified else ''

    def rows(name):
        return list(wb[name].iter_rows(values_only=True))
    r6 = rows('Figure 6')
    yrs = [int(y) for y in r6[2][1:52]]
    assert r6[7][0] == 'Agriculture' and r6[10][0] == 'LULUCF'
    ag_tot = dict(zip(yrs, r6[7][1:52]))
    r25 = rows('Figure 25')
    y25 = [int(y) for y in r25[2][1:52]]
    comm = {r[0]: dict(zip(y25, r[1:52])) for r in r25[3:13] if r[0]}
    r30 = rows('Figure 30')
    y30 = [int(y) for y in r30[2][1:52]]
    buck = {r[0]: dict(zip(y30, r[1:52])) for r in r30[3:6]}
    lu_inv = dict(zip(y30, r30[6][1:52]))
    lu_proj = dict(zip(y30, r30[7][1:52]))
    lu_tot = {y: (lu_inv[y] if lu_inv[y] is not None else lu_proj[y]) for y in y30}
    assert set(comm) == {'Grazing beef', 'Grain fed beef', 'Dairy', 'Sheep', 'Pigs', 'Crops', 'Other animals',
                         'Fertilisers', 'Lime and urea', 'Other'}, sorted(comm)
    assert set(buck) == {FC, AO, FO}, sorted(buck)
    kt = lambda d: {y: float(v) * 1000.0 for y, v in d.items() if v is not None}
    return (kt(ag_tot), {k: kt(v) for k, v in comm.items()}, kt(lu_tot), {k: kt(v) for k, v in buck.items()},
            sha(raw), modified)


# ----------------------------------------------------------------------------- assignment rules
def ag_rule(p):
    leaf = p.split(' > ')[-1]
    has = lambda *k: any(x in p for x in k)
    lv = next((k for k in LV_T1 if k in p), None)
    if has('3.A Enteric'):
        if has('Feedlot'): return ('no', '', '', 'No feedlot land use or commodity in LUTO.', 'OBSERVED (no field)')
        if has('3.A.4 Other Livestock'): return ('no', '', '', 'No LUTO land use or commodity for other livestock.', 'OBSERVED (no field)')
        if has('3.A.3 Swine'): return ('yes', '5', 'off-land pork Enteric fermentation (CH4)', 'GLEAM intensity x demand (AR6 STATED).', 'OBSERVED')
        if lv: return ('yes', '1', 'CO2E_KG_HEAD_ENTERIC', 'Head-based enteric factor on dairy/beef/sheep land uses.', 'OBSERVED')
    if has('3.B Manure'):
        if has('3.B.5 Indirect'):
            if has('Swine', 'Poultry'): return ('yes', '5', 'off-land Manure (N2O)', 'GLEAM manure N2O; whether it includes indirect N2O is UNVERIFIED.', 'INFERRED')
            if has('Feedlot'): return ('no', '', '', 'No feedlot.', 'OBSERVED (no field)')
            if has('Atmospheric Deposition'): return ('no', '', '', 'Dairy manure atmospheric deposition: no named LUTO field (V0 convention).', 'UNVERIFIED')
            return ('yes', '1', 'CO2E_KG_HEAD_IND_LEACH_RUNOFF', 'Dairy manure leaching/runoff mapped to the livestock leaching field.', 'INFERRED')
        if has('Feedlot'): return ('no', '', '', 'No feedlot.', 'OBSERVED (no field)')
        if has('3.B.3 Swine'): return ('yes', '5', 'off-land pork Manure (CH4, N2O)', 'GLEAM.', 'OBSERVED')
        if has('Poultry'):
            if has('Layers', 'Meat chicken'): return ('yes', '5', 'off-land chicken/eggs Manure (CH4, N2O)', 'GLEAM chicken and eggs.', 'OBSERVED')
            return ('no', '', '', 'Ducks and other poultry meat: no LUTO commodity (chicken and eggs only).', 'INFERRED')
        if lv: return ('yes', '1', 'CO2E_KG_HEAD_MANURE_MGT', 'Head-based manure factor.', 'OBSERVED')
        return ('no', '', '', 'Other livestock: no LUTO land use.', 'OBSERVED (no field)')
    if has('3.C Rice'): return ('no', '', '', 'Crop SOIL factor for rice is the same order as other irrigated crops: no CH4 term.', 'INFERRED')
    if has('3.D.a.1 Inorganic'):
        if leaf.endswith('Non-irrigated pasture'): return ('no', '', '', 'Dryland livestock fields carry no fertiliser N2O.', 'INFERRED')
        if leaf.endswith('Irrigated pasture'): return ('yes', '1', 'agGHG_irrpast.h5 CO2E_KG_HA_SOIL', 'Irrigated pasture SOIL field.', 'OBSERVED')
        return ('yes', '1', 'agGHG_crops.h5 CO2E_KG_HA_SOIL', 'Crop SOIL field (STATED soil N2O; residue share UNVERIFIED).', 'OBSERVED field / STATED content')
    if has('3.D.a.2'): return ('no', '', '', 'Organic fertiliser (animal waste applied, sewage sludge): no LUTO field.', 'INFERRED')
    if has('3.D.a.3 Urine'):
        if lv: return ('yes', '1', 'CO2E_KG_HEAD_DUNG_URINE', 'Head-based dung/urine factor.', 'OBSERVED')
        return ('no', '', '', 'Other livestock / poultry on pasture: no LUTO land use.', 'OBSERVED (no field)')
    if has('3.D.a.4 Crop Residue'): return ('no', '', '', 'No named residue field; may sit inside crop SOIL (open V0 question).', 'UNVERIFIED')
    if has('3.D.a.5', '3.D.a.6'): return ('no', '', '', 'No field.', 'INFERRED')
    if has('3.D.b.1 Atmospheric'): return ('no', '', '', 'Atmospheric deposition: no named LUTO field (V0 convention).', 'UNVERIFIED')
    if has('3.D.b.2 Nitrogen Leaching'):
        if has('3.D.b.2.ii Manure') and lv: return ('yes', '1', 'CO2E_KG_HEAD_IND_LEACH_RUNOFF', 'Livestock leaching/runoff field. NIR 2024 reports leaching per animal, so the V0 N-share split is not needed.', 'OBSERVED')
        return ('no', '', '', 'Fertiliser / residue / sewage / other-livestock leaching: no LUTO field.', 'INFERRED')
    if has('3.F Field'): return ('no', '', '', 'No field found.', 'UNVERIFIED')
    if has('3.G Liming'): return ('no', '', '', 'CHEM_APPL is described as non-liming energy; no liming field.', 'UNVERIFIED')
    if has('3.H Urea'): return ('no', '', '', 'No dedicated urea field.', 'UNVERIFIED')
    raise ValueError('unassigned AG leaf: ' + p)


def lulucf_rule(p):
    has = lambda *k: any(x in p for x in k)
    if has('Non-temperate fire management'):
        return ('LULUCF_EXO', 'partial', '2', 'Savanna Burning (EDS): avoided CH4/N2O + sequestered CO2 (increments only)',
                'Savanna fire. NIR 2024 marks 3.E IE and reports it under non-temperate fire management in 4.A, 4.C and 4.D. '
                'Baseline level is not in LUTO; term 2 adds an avoided-emissions delta from EDS adoption. Level is exogenous.',
                'OBSERVED (tree) / INFERRED')
    if has('4.B.2.1 Forest Land converted to Cropland', '4.C.2.1 Forest Land converted to Grassland'):
        return ('LULUCF_EXO', 'partial', '4', 'flow_ghg_ag2ag . D (natural -> modified land, clearing), increments only',
                'Forest clearing to cropland/grassland, with its post-clearing legacy emissions. Baseline clearing is exogenous '
                '(Nick decision B); LUTO term 4 stays on the LULUCF row, so any clearing LUTO chooses is additional to it, as plantings are.',
                'STATED (decision) / OBSERVED (code)')
    if has(*PLANTING_LEAVES):
        what = 'EP / riparian / agroforestry / carbon plantings' if 'Environmental' in p else 'none (commercial plantations are not a LUTO land use)'
        return ('LULUCF_EXO', 'partial' if 'Environmental' in p else 'no', '3' if 'Environmental' in p else '', what,
                'Plantings stay exogenous (Nick plantings rule): DCCEEW baseline plantings in EXO; LUTO term 3 counts only above '
                'the baseline planting rate in plantings_baseline.csv.', 'STATED (decision) / INFERRED (baseline)')
    if has('4.A.2.2.i.d Natural Regeneration'):
        return ('LULUCF_EXO', 'partial', '2 (HIR), 3 (Destocked)', 'HIR regrowth; Destocked natural land regrowth',
                'Includes ERF human-induced regeneration projects. LUTO HIR / destocked regrowth is new-from-base only; '
                'the cohort to date is exogenous; LUTO adds increments.', 'OBSERVED (code) / INFERRED (category)')
    if has('4.C.1.i Sparse woody'):
        return ('LULUCF_EXO', 'partial', '2, 3, 4', 'HIR / Destocked regrowth below forest threshold; term 4 natural->livestock-natural degradation',
                'Climate- and management-driven woody change on grassland. LUTO touches it only through new-from-base increments.', 'INFERRED')
    if has('4.C.1.ii Grassland soils'):
        return ('LULUCF_EXO', 'partial', '2', 'Ecological Grazing soil carbon (off by default)',
                'Climate-driven grassland SOC (large La Nina swings). LUTO term 2 would add increments only.', 'INFERRED')
    if has('4.B.1.1 Cropland Soils'):
        return ('LULUCF_EXO', 'partial', '2', 'Biochar soil carbon', 'Cropland SOC change; LUTO biochar SOC is an increment only.', 'INFERRED')
    return ('LULUCF_EXO', 'no', '', '', 'No LUTO term represents this category.', 'OBSERVED (no term)')


def dc_comm(p):
    leaf = p.split(' > ')[-1]
    if '3.G ' in p or '3.H ' in p: return 'Lime and urea'
    if '3.F Field' in p or '3.C Rice' in p or 'Crop Residue' in p: return 'Crops'
    if 'Fertiliser >' in p or 'Inorganic Fertilisers' in p: return 'Fertilisers'
    if 'Sewage' in p or 'Mineralisation' in p or 'Histosols' in p: return 'Other'
    for k, c in [('Feedlot', 'Grain fed beef'), ('Beef Cattle - Pasture', 'Grazing beef'), ('Dairy', 'Dairy'),
                 ('Sheep', 'Sheep'), ('Swine', 'Pigs'), ('Poultry', 'Other animals'), ('Other Livestock', 'Other animals')]:
        if k in p: return c
    if any(k in leaf for k in ['Buffalo', 'Camels', 'Deer', 'Goats', 'Horses', 'Donkeys', 'Alpacas', 'Ostriches']):
        return 'Other animals'
    raise ValueError(p)


def dc_bucket(p):
    if '4.A Forest Land' in p or '4.G Harvested' in p: return FO
    if any(k in p for k in ['4.B.1 ', '4.C.1 ', '4.D.1 ', '4.E.1 ', '4.F.1']): return AO
    if any(k in p for k in ['4.B.2 ', '4.C.2 Land', '4.D.2 ', '4.E.2 ']): return FC
    raise ValueError(p)


CODE_RE = re.compile(r'^([0-9]+(?:\.[A-Za-z0-9]+)*\.?)\s+(.*)$')


def split_code(name):
    m = CODE_RE.match(name)
    return (m.group(1).rstrip('.'), m.group(2)) if m else ('', name)


# ----------------------------------------------------------------------------- writers
def write_csv(path, header, rows):
    buf = io.StringIO(newline='')
    import csv
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(header)
    for r in rows:
        w.writerow(r)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(buf.getvalue())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='output folder for the target files (e.g. docs/rebase/targets)')
    ap.add_argument('--scope-matrix', default=None, help='path for scope_matrix.csv (default: <out>/../scope_matrix.csv)')
    ap.add_argument('--cache', default=None, help='directory to cache downloaded files in (read cache if present, else download and store)')
    ap.add_argument('--offline', action='store_true', help='fail instead of downloading on a cache miss (requires --cache)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    scope_path = a.scope_matrix or os.path.join(a.out, '..', 'scope_matrix.csv')
    global CACHE_DIR, OFFLINE
    CACHE_DIR = a.cache
    OFFLINE = a.offline

    # ---- NIR 2024 ----
    nodes, AGID, LUID, defaults_sha = anga_tree()
    vals, all_sha = anga_values(nodes.keys(), ['ALL'])
    check('anga_all_gas_canonical', all_sha)
    V = {(s, y): v for s, y, g, v in vals}
    top_ag = {split_code(nodes[i]['name'])[0]: i for i in nodes if nodes[i]['parent'] == AGID}
    gvals, gas_sha = anga_values([AGID] + list(top_ag.values()), ['CO2', 'CH4', 'N2O'])
    check('anga_by_gas_canonical', gas_sha)
    G = {(s, y, g): v for s, y, g, v in gvals}
    years = sorted({y for _, y in V})
    assert years[0] <= 2010 and years[-1] == ANCHOR, years
    kids = {}
    for i, n in nodes.items():
        if n['parent'] is not None:
            kids.setdefault(n['parent'], []).append(i)
    has = lambda i: any((i, y) in V for y in years)

    def path(i):
        p = []
        while i is not None:
            p.append(nodes[i]['name'])
            i = nodes[i]['parent']
        return ' > '.join(reversed(p))

    def sector(i):
        while nodes[i]['parent'] is not None:
            i = nodes[i]['parent']
        return '3' if i == AGID else '4'
    leaves = sorted(i for i in nodes if has(i) and not any(has(c) for c in kids.get(i, [])))
    for y in years:
        for sid, sec in ((AGID, '3'), (LUID, '4')):
            d = math.fsum(V.get((i, y), 0.0) for i in leaves if sector(i) == sec) - V[(sid, y)]
            if abs(d) > 1e-6:
                raise SystemExit(f'leaf sum check failed sector {sec} year {y}: {d}')

    # ---- scope matrix ----
    S = []
    for i in leaves:
        p = path(i)
        code, name = split_code(nodes[i]['name'])
        pcode = split_code(nodes[nodes[i]['parent']]['name'])[0] if nodes[i]['parent'] is not None else ''
        if sector(i) == '3':
            mod, term, comp, why, lab = ag_rule(p)
            S.append(dict(node_id=i, category_code=code, name=name, parent=pcode, nir_path=p, value_2024_kt=V.get((i, ANCHOR), 0.0),
                          assignment='AG', modelled_by_luto=mod, flag='' if mod == 'yes' else 'AG, not modelled by LUTO today',
                          luto_term=term, luto_component=comp, partial='no' if mod == 'yes' else 'n/a (not modelled)',
                          split_rule='none (NIR leaf is atomic)',
                          reasoning='Whole sector 3 is the agriculture target. ' + why, label=lab, dcceew_row=dc_comm(p)))
        else:
            asg, part, term, comp, why, lab = lulucf_rule(p)
            S.append(dict(node_id=i, category_code=code, name=name, parent=pcode, nir_path=p, value_2024_kt=V.get((i, ANCHOR), 0.0),
                          assignment=asg, modelled_by_luto=part, flag='', luto_term=term, luto_component=comp, partial=part,
                          split_rule='none applied (whole leaf)', reasoning=why, label=lab, dcceew_row=dc_bucket(p)))
    idx = {r['node_id']: r for r in S}
    cols = list(S[0].keys())
    write_csv(scope_path, cols, [[fmt(r[c]) if c == 'value_2024_kt' else r[c] for c in cols]
                                 for r in sorted(S, key=lambda r: (r['nir_path'], r['node_id']))])

    def ssum(pred, y):
        return math.fsum(V.get((i, y), 0.0) for i in leaves if pred(idx[i]))
    H = {k: {y: ssum(lambda r, k=k: r['assignment'] == k, y) for y in years} for k in ('AG', 'LULUCF_MOD', 'LULUCF_EXO')}
    NIR3 = {y: V[(AGID, y)] for y in years}
    NIR4 = {y: V[(LUID, y)] for y in years}
    comm_nir = {c: {y: ssum(lambda r, c=c: r['assignment'] == 'AG' and r['dcceew_row'] == c, y) for y in years}
                for c in ['Crops', 'Dairy', 'Fertilisers', 'Grain fed beef', 'Grazing beef', 'Lime and urea', 'Other',
                          'Other animals', 'Pigs', 'Sheep']}
    buck_nir = {b: {y: ssum(lambda r, b=b: r['assignment'] != 'AG' and r['dcceew_row'] == b, y) for y in years} for b in (FC, AO, FO)}

    # ---- DCCEEW 2025 ----
    DA, DC, DL, DB, xlsx_sha, xlsx_modified = dcceew_tables()

    def off(n24, d, y): return n24 + (d[y] - d[ANCHOR])

    def conv(n24, d, y):
        if y >= CONVERGE_TO: return d[y]
        w = (y - ANCHOR) / (CONVERGE_TO - ANCHOR)
        return n24 * (1 - w) + d[CONVERGE_TO] * w
    ag_comm_p = {c: {y: off(comm_nir[c][ANCHOR], DC[c], y) for y in YR_PROJ} for c in comm_nir}
    AGP = {y: math.fsum(ag_comm_p[c][y] for c in comm_nir) for y in YR_PROJ}

    # agriculture LUTO does not model: exogenous on the AG row. History: the sector 3 leaves not modelled by LUTO.
    # Projection: each DCCEEW commodity keeps its FY2024 unmodelled share of its spliced projection.
    unmod = lambda r: r['assignment'] == 'AG' and r['modelled_by_luto'] != 'yes'
    AGX_h = {y: ssum(unmod, y) for y in years}
    agx_share = {c: (ssum(lambda r, c=c: unmod(r) and r['dcceew_row'] == c, ANCHOR) / comm_nir[c][ANCHOR]
                     if comm_nir[c][ANCHOR] else 0.0) for c in comm_nir}
    AGXp = {y: math.fsum(agx_share[c] * ag_comm_p[c][y] for c in comm_nir) for y in YR_PROJ}
    FOp = {y: off(buck_nir[FO][ANCHOR], DB[FO], y) for y in YR_PROJ}
    AOp = {y: conv(buck_nir[AO][ANCHOR], DB[AO], y) for y in YR_PROJ}
    FCp = {y: conv(buck_nir[FC][ANCHOR], DB[FC], y) for y in YR_PROJ}
    w0, w1 = BUDGET_WINDOW
    share = math.fsum(H['LULUCF_MOD'][y] for y in range(w0, w1 + 1)) / math.fsum(buck_nir[FC][y] for y in range(w0, w1 + 1))
    MODp = {y: share * FCp[y] for y in YR_PROJ}
    EXOp = {y: math.fsum([FOp[y], AOp[y], (1 - share) * FCp[y]]) for y in YR_PROJ}
    LUp = {y: math.fsum([FOp[y], AOp[y], FCp[y]]) for y in YR_PROJ}

    # ---- series ----
    def role(y):
        if y < BASE_YEAR: return 'REFERENCE (pre-base)'
        if y == BASE_YEAR: return 'BASE'
        if y <= ANCHOR: return 'BENCHMARK'
        return 'BINDING'
    method = {
        'AG': ('sum of all NIR 2024 sector 3 leaves',
               'NIR 2024 FY2024 level + DCCEEW 2025 year-on-year change from FY2024, by commodity (offset splice)'),
        'AG_EXO': ('sum of NIR 2024 sector 3 leaves not modelled by LUTO (scope_matrix modelled_by_luto != yes); part of AG',
                   'AG projection by commodity x the FY2024 unmodelled share of that commodity (proportional share); part of AG'),
        'LULUCF_MOD': ('sum of NIR 2024 leaves assigned LULUCF_MOD (none: direct clearing is exogenous, Nick decision B)',
                       f'forest-conversion bucket (converge to DCCEEW level by FY2030) x MOD share of that bucket FY2021-2024 = {share:.6f}'),
        'LULUCF_EXO': ('sum of NIR 2024 leaves assigned LULUCF_EXO (all plantings and all direct clearing)',
                       'Forests bucket (offset) + Agricultural and other land (converge by FY2030) + (1 - MOD share) x forest-conversion bucket (converge)'),
    }
    series_vals = {'AG': (H['AG'], AGP), 'AG_EXO': (AGX_h, AGXp), 'LULUCF_MOD': (H['LULUCF_MOD'], MODp), 'LULUCF_EXO': (H['LULUCF_EXO'], EXOp)}
    hdr = ['year_end', 'fy_label', 'luto_year', 'value_t_co2e', 'value_kt_co2e_ar5', 'source', 'status', 'method', 'role']
    full = {}
    for k, (hv, pv) in series_vals.items():
        rows, full[k] = [], {}
        for y in YR_HIST:
            full[k][y] = hv[y]
            rows.append([y, fy(y), y, fmt(hv[y] * 1000, 3), fmt(hv[y]), 'NIR2024', 'HISTORY', method[k][0], role(y)])
        for y in YR_PROJ:
            full[k][y] = pv[y]
            rows.append([y, fy(y), y, fmt(pv[y] * 1000, 3), fmt(pv[y]), 'DCCEEW2025', 'PROJECTION', method[k][1], role(y)])
        for y in YR_HELD:
            full[k][y] = pv[2040]
            rows.append([y, fy(y), y, fmt(pv[2040] * 1000, 3), fmt(pv[2040]), 'DCCEEW2025', 'HELD (FY2040 value, not projection)',
                         'held flat at FY2040 (Nick: "Hold flat at 2040")', 'BINDING'])
        fn = {'AG': 'series_agriculture.csv', 'AG_EXO': 'series_agriculture_exogenous.csv', 'LULUCF_MOD': 'series_lulucf_modelled.csv',
              'LULUCF_EXO': 'series_lulucf_exogenous.csv'}[k]
        write_csv(os.path.join(a.out, fn), hdr, rows)

    # ---- components long ----
    comp = []
    for i in leaves:
        r = idx[i]
        for y in YR_HIST:
            comp.append([y, r['assignment'], f'NIR:{i}', r['nir_path'], fmt(V.get((i, y), 0.0)), 'NIR2024', 'HISTORY'])
    for y in YR_PROJ + YR_HELD:
        yy = min(y, 2040)
        st = 'PROJECTION' if y <= 2040 else 'HELD'
        for c in sorted(comm_nir):
            comp.append([y, 'AG', f'DCCEEW:Figure25:{c}', f'DCCEEW commodity {c} (offset)', fmt(ag_comm_p[c][yy]), 'DCCEEW2025', st])
        comp.append([y, 'LULUCF_EXO', 'DCCEEW:Figure30:Forests', 'DCCEEW bucket Forests (offset); includes all plantings', fmt(FOp[yy]), 'DCCEEW2025', st])
        comp.append([y, 'LULUCF_EXO', 'DCCEEW:Figure30:AgriculturalOther', 'DCCEEW bucket Agricultural and other land (converge by FY2030)', fmt(AOp[yy]), 'DCCEEW2025', st])
        comp.append([y, 'LULUCF_EXO', 'DCCEEW:Figure30:ForestConversion:EXO', 'forest-conversion bucket x (1 - MOD share)', fmt((1 - share) * FCp[yy]), 'DCCEEW2025', st])
        comp.append([y, 'LULUCF_MOD', 'DCCEEW:Figure30:ForestConversion:MOD', 'forest-conversion bucket x MOD share', fmt(share * FCp[yy]), 'DCCEEW2025', st])
    comp.sort(key=lambda r: (r[0], r[1], r[2]))
    write_csv(os.path.join(a.out, 'series_components_long.csv'),
              ['year_end', 'series', 'component_id', 'component', 'value_kt_co2e_ar5', 'source', 'status'], comp)

    # AG_EXO is a part of AG, not a fourth series: it never enters the sum check below
    for y in full['AG']:
        if not 0.0 <= full['AG_EXO'][y] <= full['AG'][y]:
            raise SystemExit(f'AG_EXO outside [0, AG] in {y}: {full["AG_EXO"][y]} vs {full["AG"][y]}')

    # ---- sum check ----
    sc = []
    for y in YR_HIST + YR_PROJ + YR_HELD:
        s3 = math.fsum(full[k][y] for k in ('AG', 'LULUCF_MOD', 'LULUCF_EXO'))
        if y <= ANCHOR:
            ref, rdef, raw = NIR3[y] + NIR4[y], 'NIR 2024 sector 3 + sector 4', ''
        elif y <= 2040:
            ref, rdef, raw = AGP[y] + LUp[y], 'spliced DCCEEW sector totals (AG offset + LULUCF hybrid)', fmt(DA[y] + DL[y])
        else:
            ref, rdef, raw = AGP[2040] + LUp[2040], 'FY2040 spliced total (HELD)', ''
        d = s3 - ref
        if abs(d) > 1e-6:
            raise SystemExit(f'sum check failed {y}: {d}')
        held_ok = '' if y <= 2040 else str(all(full[k][y] == full[k][2040] for k in full))
        sc.append([y, fy(y), fmt(full['AG'][y]), fmt(full['LULUCF_MOD'][y]), fmt(full['LULUCF_EXO'][y]), fmt(s3), fmt(ref), rdef,
                   fmt(d, 9), 'True', raw, held_ok])
    write_csv(os.path.join(a.out, 'sum_check.csv'),
              ['year_end', 'fy_label', 'ag_kt', 'lulucf_mod_kt', 'lulucf_exo_kt', 'sum_three_kt', 'reference_kt', 'reference_def',
               'diff_kt', 'pass', 'dcceew_raw_sector_sum_kt', 'held_equals_fy2040'], sc)

    # ---- tolerances and budget (PROPOSAL) ----
    tol = []
    for y in YR_HIST:
        half = math.sqrt(math.fsum((G.get((top_ag[c], y, g), 0.0) * u) ** 2 for c, g, u in UNC_AG))
        tol.append(['AG', 'annual, history year', str(y), fmt(NIR3[y], 3), fmt(100 * half / NIR3[y], 3), fmt(half, 3),
                    'Approach 1 propagation (independent) of NIR 2024 Vol 2 Annex II Table A2.3 (printed p.19) level uncertainties by category and gas', 'PROPOSAL'])
    bud = math.fsum(H['LULUCF_MOD'][y] for y in range(w0, w1 + 1))
    unc = math.sqrt(math.fsum((UNC_FOREST_CONVERSION * H['LULUCF_MOD'][y]) ** 2 for y in range(w0, w1 + 1)))
    tol.append(['LULUCF_MOD_BUDGET', 'cumulative window FY2021-FY2024', '2021-2024', fmt(bud, 3), fmt(100 * UNC_FOREST_CONVERSION, 3),
                fmt(UNC_FOREST_CONVERSION * bud, 3), 'NIR 2024 Vol 2 Annex II Table A2.4 (printed p.21) B.2/C.2 +/-27.9%, fully correlated across years (lean)', 'PROPOSAL'])
    tol.append(['LULUCF_MOD_BUDGET', 'cumulative window FY2021-FY2024 (alternative: independent years)', '2021-2024', fmt(bud, 3),
                fmt(100 * unc / bud, 3) if bud else '', fmt(unc, 3), 'as above, quadrature over the four years', 'PROPOSAL'])
    write_csv(os.path.join(a.out, 'tolerances.csv'),
              ['series', 'applies_to', 'year_end', 'reference_kt', 'tolerance_pct', 'tolerance_kt', 'basis', 'status'], tol)
    write_csv(os.path.join(a.out, 'budget.csv'),
              ['series', 'window_start_year_end', 'window_end_year_end', 'window_luto_years', 'window_solve_years', 'budget_kt',
               'tolerance_kt', 'by_year_kt', 'rule', 'status'],
              [['LULUCF_MOD', w0, w1, '2021-2024', '2022-2024 (2021 is the base year)', fmt(bud, 3), fmt(UNC_FOREST_CONVERSION * bud, 3),
                '; '.join(f'{y}: {fmt(H["LULUCF_MOD"][y], 3)}' for y in range(w0, w1 + 1)),
                'sum over window of LUTO modelled LULUCF within budget_kt +/- tolerance_kt', 'PROPOSAL']])

    # ---- plantings baseline (INFERRED / PROPOSAL) ----
    pl = {p: next(i for i in leaves if nodes[i]['name'] == p) for p in PLANTING_LEAVES}
    prow = []
    for label, keys in (('environmental plantings (4.A.2.2.i.c)', [PLANTING_LEAVES[2]]),
                        ('all plantings (4.A.2.2.i.a + i.b + i.c)', list(PLANTING_LEAVES))):
        R = {y: math.fsum(V.get((pl[k], y), 0.0) for k in keys) for y in range(2020, ANCHOR + 1)}
        dlt = {y: R[y] - R[y - 1] for y in range(w0, w1 + 1)}
        avg_d = math.fsum(dlt.values()) / len(dlt)
        avg_l = math.fsum(R[y] for y in range(w0, w1 + 1)) / len(dlt)
        lean = 'LEAN' if keys == [PLANTING_LEAVES[2]] else 'alternative'
        prow.append([label, 'annual change in net removals (kt CO2-e / yr per yr)', fmt(avg_d, 3),
                     '; '.join(f'{y}: {fmt(dlt[y], 3)}' for y in range(w0, w1 + 1)),
                     'mean of year-on-year change FY2021-FY2024 = (R2024 - R2020)/4; negative = removals growing',
                     'constant from base year 2021', f'INFERRED / PROPOSAL ({lean})'])
        prow.append([label, 'mean annual net removals level (kt CO2-e / yr)', fmt(avg_l, 3),
                     '; '.join(f'{y}: {fmt(R[y], 3)}' for y in range(w0, w1 + 1)), 'mean FY2021-FY2024 (context only)',
                     'n/a', 'INFERRED (context)'])
    prow.append(['planted area (ha / yr)', 'annual new planted area', '', '',
                 'NOT AVAILABLE: the NIR 2024 LULUCF activity table (Tables 1a-17) has no plantation or environmental-planting area series',
                 'n/a', 'UNVERIFIED (no public series found)'])
    write_csv(os.path.join(a.out, 'plantings_baseline.csv'),
              ['quantity', 'definition', 'value', 'by_year', 'basis', 'carried_forward', 'status'], prow)

    # ---- agriculture not modelled (history of series_agriculture_exogenous.csv, by modelled / not modelled) ----
    ag_nm = [[y, fmt(ssum(lambda r: r['assignment'] == 'AG' and r['modelled_by_luto'] != 'yes', y)),
              fmt(ssum(lambda r: r['assignment'] == 'AG' and r['modelled_by_luto'] == 'yes', y)), fmt(NIR3[y]), 'NIR2024',
              'INFO (history of series_agriculture_exogenous.csv)']
             for y in YR_HIST]
    write_csv(os.path.join(a.out, 'ag_not_modelled.csv'),
              ['year_end', 'ag_not_modelled_kt', 'ag_modelled_kt', 'ag_total_kt', 'source', 'status'], ag_nm)

    # ---- provenance (last; hashes the other outputs) ----
    scope_in_out = os.path.abspath(os.path.dirname(scope_path)) == os.path.abspath(a.out)   # hashed once, by its own row below
    outs = sorted(f for f in os.listdir(a.out) if f.endswith('.csv') and f != 'provenance.csv'
                  and not (scope_in_out and f == os.path.basename(scope_path)))
    me = open(os.path.abspath(__file__), 'rb').read()
    P = [
        ['source', 'NIR 2024 sectors 3 and 4 (ANGA API OutputTimeSeries, gas ALL)', ANGA_BASE + 'OutputTimeSeries', all_sha,
         'sha256 of canonical sorted values', SOURCE_DATES['NIR2024'], 'AR5 GWP100'],
        ['source', 'NIR 2024 agriculture top categories by gas (ANGA API)', ANGA_BASE + 'OutputTimeSeries', gas_sha,
         'sha256 of canonical sorted values', SOURCE_DATES['NIR2024'], 'AR5 GWP100'],
        ['source', 'ANGA sector tree (Defaults, sectortreeParis)', ANGA_DEFAULTS_URL, defaults_sha, 'sha256 of raw response',
         SOURCE_DATES['NIR2024'], ''],
        ['source', "DCCEEW Australia's emissions projections 2025 chart data (Figures 6, 25, 30; Baseline scenario)", DCCEEW_XLSX_URL,
         xlsx_sha, 'sha256 of raw file', f'{xlsx_modified} (workbook modified property)', 'AR5 GWP100 (report fn 55)'],
        ['script', 'luto/tools/ghg_targets.py', '', sha(me), 'sha256 of script', '', ''],
    ]
    P += [['decision', f'Nick decision {n + 1} (verbatim)', '', '', d, '', ''] for n, d in enumerate(NICK_DECISIONS)]
    P += [['decided', 'LULUCF_MOD definition', '', '', 'no NIR leaf: direct clearing is exogenous (Nick decision B); LUTO term 4 is additional', '', ''],
          ['decided', 'LULUCF projection MOD share', '', '', f'{share:.6f} of the forest-conversion bucket, FY2021-2024', '', ''],
          ['decided', 'unmodelled agriculture', '', '', 'series_agriculture_exogenous.csv, a part of AG on the AG row left-hand side; '
           'projection keeps each commodity\'s FY2024 unmodelled share', '', ''],
          ['provisional', 'tolerances', '', '', 'tolerances.csv, PROPOSAL (open)', '', ''],
          ['provisional', 'plantings baseline', '', '', 'plantings_baseline.csv, INFERRED / PROPOSAL', '', ''],
          ['year_basis', 'LUTO year Y = FY ending Y; base year 2021 (NLUM 2020-21)', '', '', '', '', '']]
    P += [['output', f, '', sha(open(os.path.join(a.out, f), 'rb').read()), 'sha256', '', ''] for f in outs]
    P += [['output', os.path.basename(scope_path), '', sha(open(scope_path, 'rb').read()), 'sha256', '', '']]
    write_csv(os.path.join(a.out, 'provenance.csv'),
              ['kind', 'item', 'url_or_endpoint', 'sha256', 'note', 'source_date', 'gwp'], P)
    print(f'wrote {len(outs) + 2} files to {a.out}; MOD share {share:.6f}; budget {bud:.3f} kt')


if __name__ == '__main__':
    main()
