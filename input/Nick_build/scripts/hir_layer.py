#!/usr/bin/env python
"""Build LUTO's HIR regrowth rate layer (hir_rate_layer_gross.h5): observed CER issuance per CEA hectare, grossed up, by NRM region.

Sources (public, downloaded at run time, sha256-checked against pinned values):
  - Clean Energy Regulator (CER) ACCU Scheme project register (CSV): HIR method projects, registration dates, nominated
    permanence period, ACCU issuance by financial year. The register is a live file that CER updates without a version
    stamp: the raw file is byte-pinned, the cached copy is the reproduction source, and --live-check fetches the
    live file and stops if it differs from the pin.
Input from another tool: the ag_mgt_caps output folder (ag_mgt_caps.py --base-year 2021, input/Nick_build/ag_mgt_caps_2021):
hir_project_cells.csv (CEA coverage fraction per NLUM cell and project) and hir_projects.csv (mapped flag, registration FY).
Its provenance.csv must record the same register pin. The project cells depend on the land-use map the caps tool read
(base-year switch).
Inherited 2010-build inputs read from --input (inherited_2010/, copies of 2010-build files), sha256 recorded in provenance: REGION_NRM_r.h5, REGION_STATE_r.h5,
real_area.h5, NLUM_2010-11_mask.tif (cell count check), ag_landuses.csv (land-use ids), and the lumap.h5 given by
--lumap, which must be the map of the run's base year (--base-year): the 2010 map or the 2021 map of the 2021-base tools
(livestock_map.py output), each pinned by dataset content sha256, since HDF5 file bytes vary between runs.
The output is therefore not built from public sources alone.

Method:
  1. Per HIR project: mean issuance FY2020-FY2024 over the years from max(2020, registration FY) to 2024 (zeros after
     registration kept); gross = issuance / (1 - buffer - 25-year discount if the nominated permanence is 25 years), the
     additive unit entitlement of the Carbon Credits (Carbon Farming Initiative) Act 2011 s 16(2) (0.75 for 25-year
     projects, 0.95 for 100-year projects; each project's own nominated permanence period from the register).
  2. CEA ha = sum over the project's cells of fraction x real_area. A project goes to the NRM holding most of its CEA ha;
     the state of an NRM is its modal state by cell area. Projects that are mapped and have CEA ha > 0 enter the rates.
  3. Region rate = sum of gross mean issuance / sum of CEA ha; regions with fewer than 3 projects take their state's pooled
     rate. The issuance-basis rate is built the same way from mean issuance. The FY2020-FY2024 window and the
     fewer-than-3 pooling rule are analysis choices, no published source.
  4. HIR_RATE (gross) and HIR_RATE_ISSUANCE on every cell of an NRM with a rated project, 0 elsewhere; HIR_ELIGIBLE 1/0.
  The project table is written once and read back before the region step, so the rates use the 9-significant-digit values
  of hir_projects_rates.csv.
  The ERF site-potential M layer (HIR_RATE_M) is dropped: with M all-NaN the gross layer is bit-equal.
Outputs in --out: hir_rate_layer_gross.h5 (key hir_rate_layer_gross, pandas table, zlib 9, float32 HIR_RATE, HIR_ELIGIBLE,
HIR_RATE_ISSUANCE, int64 index 0..N-1); hir_projects_rates.csv;
hir_region_rates_gross.csv; hir_region_rates.csv and hir_state_rates.csv (issuance basis);
hir_eligible_beef_sheep_natural_by_state.csv (on the --lumap map); hir_rate_layer_gross.nc (NetCDF copy in the jinzhu
layout, one variable per column, nc_layout.py); provenance.csv (last). The map enters only that
eligibility report; the rate layer is per NRM region.

Run as a plain script from the repository root (no luto imports):

    python input/Nick_build/scripts/hir_layer.py --input input/Nick_build/inherited_2010 --lumap input/Nick_build/livestock_map/lumap.h5 --base-year 2021
        --caps input/Nick_build/ag_mgt_caps_2021 --out input/Nick_build/hir_layer --cache input/Nick_build/_cache [--offline] [--live-check] [--source NAME=PATH]

Dependencies: numpy, pandas, PyTables, rasterio, xarray, netCDF4. No luto imports (nc_layout.py is a sibling module).
Memory: peak resident memory about 2.4 GB on the full grid (measured October 2026); run on a machine with at least 8
GB of RAM.
Output values and the CSV files are deterministic; HDF5 file bytes carry a group timestamp, so provenance records dataset
content hashes.
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
from importlib import metadata

import numpy as np
import pandas as pd
import rasterio

import nc_layout as NC

# ----------------------------------------------------------------------------- sources and pins
SOURCES = {   # name: (url, sha256 of the raw file, note)
    'cer_register': ('https://cer.gov.au/document/accu-scheme-project-register-0',
                     '6a10f1b5e162f6b864d204ff63ba5a753d65843a0054d83be2ece7fce6bc3ce4',
                     'CER ACCU Scheme project register (CSV), retrieved 2026-09-28; live file, raw file pinned'),
}
INHERITED = ['REGION_NRM_r.h5', 'REGION_STATE_r.h5', 'real_area.h5', 'NLUM_2010-11_mask.tif', 'ag_landuses.csv']
# the 2010 build's input/lumap.h5 (file sha256 f5347369...), by dataset content: content_sha of its values
LUMAP_2010_CONTENT_SHA = '615a598c8ca8bde4c2b42fe520170d970b0e7c2ebd9a77e31d5ada175c46b960'
# the 2021 map of the 2021-base tools (livestock_map.py lumap.h5), by dataset content: content_sha of its values, as its
# provenance.csv records it ("lumap.h5:/lumap values")
LUMAP_2021_CONTENT_SHA = '075aae2ba6d6e2f8bd928ca59264a31e69d575855e3bff4fe75611250acc61f2'
CAPS_FILES = ['hir_project_cells.csv', 'hir_projects.csv', 'provenance.csv']
NCELL = 6956407

# ============================================================================= reference table (hand-entered)
# Rule constants of the method; changing any is a ruling.
REF = {
    # HIR projects in the register: Method contains this text (all three compilations of the 2013 determination)
    'hir_method': 'Human-Induced Regeneration',
    # Issuance window FY2020-FY2024 (FY = year ending June; 2023/24 = FY2024), mean over the years since registration
    # (registration year onwards, zeros after registration kept). Analysis choice, no published source.
    'fys': [2020, 2021, 2022, 2023, 2024],
    # Risk of reversal buffer 5 % and permanence period discount 20 % for a 25-year project, both percentages of the
    # net abatement number and subtracted from it: Carbon Credits (Carbon Farming Initiative) Act 2011 s 16(2),
    # compilation 24 (C2024C00490, 14/10/2024, legislation.gov.au), PDF sha256
    # 756461a09d8b1caf6d6a63f9fc54d1c7c6972b764569a62cd075b58c2fea7d5e; CER 'Permanence obligations' page
    # (cer.gov.au, retrieved 2026-10-10), sha256 15ce9212fc27bb3080d85d334052d95757112eebf53ba8ea81415a79c6292cb4:
    # 10,000 t net abatement gives 7,500 ACCUs for a 25-year project
    'buffer': 0.05, 'perm25_discount': 0.20, 'perm25_years': 25,
    # Regions with fewer than this many rated projects take their state's pooled rate. Analysis choice, no published
    # source.
    'min_projects_region': 3,
    # ag_landuses.csv rows (0-based, no header) of Beef - natural land and Sheep - natural land (checked by name below)
    'beef_nat': 2, 'sheep_nat': 15,
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
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (luto-hir-layer/1.0)'})
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
            raise SystemExit(f'{n}: live file {url} has sha256 {got}, pinned {pin}; the source has changed')


# ----------------------------------------------------------------------------- method
def num(s):
    return pd.to_numeric(s.fillna('0').str.replace(',', '', regex=False).str.strip(), errors='coerce').fillna(0)


def fy_end(s):
    d = [int(x) for x in s.split('/')]
    return d[2] + 1 if d[1] >= 7 else d[2]


def projects(reg_path, proj_caps, cells, nrm, st, ra):
    """Per-project table and the NRM -> state map."""
    fys = REF['fys']
    reg = pd.read_csv(reg_path, encoding='cp1252', dtype=str)
    h = reg[reg['Method'].str.contains(REF['hir_method'], case=False, na=False)].copy()
    cols = {}
    for c in reg.columns:
        m = re.search(r'Units Issued in Financial Year (\d{4})/(\d{2})', c)
        if m:
            cols.setdefault(int(m.group(1)) + 1, []).append(c)
    t = pd.DataFrame({'project_id': h['Project ID'].str.strip().to_numpy()})
    for fy in fys:
        t[f'iss_FY{fy}'] = sum(num(h[c]) for c in cols[fy]).to_numpy()
    t['fy_registered'] = h['Date Project Registered'].map(fy_end).to_numpy()
    t['permanence_yr'] = pd.to_numeric(h['Nominated Permanence Period, if applicable']).to_numpy()
    t['register_state'] = h['Project location'].str.strip().to_numpy()
    t['revoked'] = h['Project revoked'].str.strip().eq('Yes').to_numpy()
    p = proj_caps.set_index('project_id')
    if not (p.fy_registered.reindex(t.project_id).to_numpy() == t.fy_registered.to_numpy()).all():
        raise SystemExit('hir_projects.csv (caps tool) and the register disagree on registration FY: different register')
    t['mapped'] = t.project_id.map(p.mapped)
    t['cea_ha_file'] = t.project_id.map(p.cea_area_ha)
    t['n_years'] = [len([y for y in fys if y >= f]) for f in t.fy_registered]
    t['iss_mean_FY20_24'] = t[[f'iss_FY{y}' for y in fys]].sum(axis=1) / t.n_years
    pre = sum(t.loc[t.fy_registered > y, f'iss_FY{y}'].sum() for y in fys)
    assert pre == 0, f'issuance before registration: {pre}'
    t['gross_factor'] = 1 - REF['buffer'] - np.where(t.permanence_yr == REF['perm25_years'], REF['perm25_discount'], 0.0)
    t['gross_mean_FY20_24'] = t.iss_mean_FY20_24 / t.gross_factor
    t['gross_FY24'] = t.iss_FY2024 / t.gross_factor
    c = cells.copy()
    c['ha'] = c.fraction * ra[c.cell]
    c['nrm'] = nrm[c.cell]
    c['state'] = st[c.cell]
    t['cea_ha'] = t.project_id.map(c.groupby('project_id').ha.sum()).fillna(0)
    byn = c.groupby(['project_id', 'nrm']).ha.sum().reset_index().sort_values(['project_id', 'ha'], ascending=[True, False])
    top = byn.drop_duplicates('project_id').set_index('project_id')
    t['nrm'] = t.project_id.map(top.nrm)
    t['nrm_ha_share'] = t.project_id.map(top.ha) / t.cea_ha
    ns = (pd.DataFrame({'n': nrm, 's': st, 'ha': ra}).groupby(['n', 's']).ha.sum().reset_index()
          .sort_values(['n', 'ha'], ascending=[True, False]).drop_duplicates('n').set_index('n').s)
    t['state'] = t.nrm.map(ns)
    t['in_rates'] = t.mapped & (t.cea_ha > 0)
    t['rate_mean'] = np.where(t.in_rates, t.iss_mean_FY20_24 / t.cea_ha.where(t.cea_ha > 0), np.nan)
    t['rate_FY24'] = np.where(t.in_rates, t.iss_FY2024 / t.cea_ha.where(t.cea_ha > 0), np.nan)
    t['rate_gross_mean'] = np.where(t.in_rates, t.gross_mean_FY20_24 / t.cea_ha.where(t.cea_ha > 0), np.nan)
    return t, ns


def region_rates(t, ns):
    """Issuance-basis and gross region and state rates from the read-back table."""
    r = t[t.in_rates].copy()
    small = REF['min_projects_region']
    g = r.groupby('nrm').agg(n_projects=('project_id', 'size'), cea_ha=('cea_ha', 'sum'), iss_mean=('iss_mean_FY20_24', 'sum'),
                             iss_FY24=('iss_FY2024', 'sum'), gross_mean=('gross_mean_FY20_24', 'sum'))
    g['state'] = g.index.map(ns)
    sg = r.groupby('state').agg(n_projects=('project_id', 'size'), cea_ha=('cea_ha', 'sum'), iss_mean=('iss_mean_FY20_24', 'sum'),
                                iss_FY24=('iss_FY2024', 'sum'), gross_mean=('gross_mean_FY20_24', 'sum'))
    sg['rate_state'] = sg.iss_mean / sg.cea_ha
    g['rate_region_own'] = g.iss_mean / g.cea_ha
    g['rate_state'] = g.state.map(sg.rate_state)
    g['pooled_to_state'] = g.n_projects < small
    g['HIR_RATE'] = np.where(g.pooled_to_state, g.rate_state, g.rate_region_own)

    gg = r.groupby('nrm').agg(n_projects=('project_id', 'size'), cea_ha=('cea_ha', 'sum'), iss_mean=('iss_mean_FY20_24', 'sum'),
                              gross_mean=('gross_mean_FY20_24', 'sum'))
    gg['state'] = gg.index.map(ns)
    sgg = r.groupby('state').agg(gross_mean=('gross_mean_FY20_24', 'sum'), cea_ha=('cea_ha', 'sum'))
    sgg['rate'] = sgg.gross_mean / sgg.cea_ha
    gg['gross_rate_own'] = gg.gross_mean / gg.cea_ha
    gg['state_pooled_gross_rate'] = gg.state.map(sgg.rate)
    gg['pooled_to_state'] = gg.n_projects < small
    gg['HIR_RATE_gross'] = np.where(gg.pooled_to_state, gg.state_pooled_gross_rate, gg.gross_rate_own)
    return g, sg, gg


# ----------------------------------------------------------------------------- provenance
def content_sha(a):
    a = np.ascontiguousarray(a)
    return sha(f'{a.dtype.str}|{a.shape}|'.encode() + a.tobytes())


def pkg_versions():
    out = {}
    for p in ('numpy', 'pandas', 'tables', 'rasterio', 'xarray', 'netCDF4'):
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
    ap.add_argument('--lumap', required=True, help="the lumap.h5 of the run's base year (refused unless it is that map)")
    ap.add_argument('--base-year', type=int, choices=(2010, 2021), default=2010,
                    help='base year of the run: 2010 (the 2010 map) or 2021 (the 2021 map); each by content sha256')
    ap.add_argument('--caps', required=True, help='output folder of ag_mgt_caps.py (e.g. input/ag_mgt_caps)')
    ap.add_argument('--out', required=True, help='output folder')
    ap.add_argument('--cache', required=True, help='shared raw-download cache (content-addressed by sha256)')
    ap.add_argument('--offline', action='store_true', help='fail instead of downloading on a cache miss')
    ap.add_argument('--live-check', action='store_true', help='also fetch the live register and stop if it differs from its pin')
    ap.add_argument('--source', action='append', default=[], metavar='NAME=PATH',
                    help='a local copy of a pinned source (must match its pin); NAME in ' + ', '.join(SOURCES))
    a = ap.parse_args()
    t0 = time.time()
    overrides = dict(s.split('=', 1) for s in a.source)
    assert set(overrides) <= set(SOURCES), set(overrides) - set(SOURCES)
    lumap_sha = content_sha(pd.read_hdf(a.lumap).to_numpy())
    pin = LUMAP_2010_CONTENT_SHA if a.base_year == 2010 else LUMAP_2021_CONTENT_SHA
    if lumap_sha != pin:
        raise SystemExit(f'{a.lumap} has content sha256 {lumap_sha}, not the {a.base_year} map ({pin}); refusing to run')
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    src = {k: fetch(k, a.cache, a.offline, overrides) for k in SOURCES}
    if a.live_check:
        live_check(list(SOURCES))
    inp = {f: os.path.join(a.input, f) for f in INHERITED}
    upstream = {f: os.path.join(a.caps, f) for f in CAPS_FILES}
    in_sha = {f: sha_file(p) for f, p in {**inp, **upstream}.items()}
    with open(upstream['provenance.csv'], encoding='utf-8', newline='') as f:
        caps_reg = [r['sha256'] for r in csv.DictReader(f) if r['url_or_endpoint'] == SOURCES['cer_register'][0]]
    if caps_reg != [SOURCES['cer_register'][1]]:
        raise SystemExit(f'the caps output was built from register {caps_reg}, not the pinned {SOURCES["cer_register"][1]}')

    # ---- grid
    with rasterio.open(inp['NLUM_2010-11_mask.tif']) as r:
        assert int((r.read(1) == 1).sum()) == NCELL
    nrm = pd.read_hdf(inp['REGION_NRM_r.h5'])['NRM_NAME'].astype(str).to_numpy()
    st = pd.read_hdf(inp['REGION_STATE_r.h5'])['STE_NAME11'].astype(str).to_numpy()
    ra = pd.read_hdf(inp['real_area.h5']).to_numpy().astype(np.float64)
    lu = pd.read_hdf(a.lumap).to_numpy()
    ag_lus = pd.read_csv(inp['ag_landuses.csv'], header=None)[0].str.strip().tolist()
    assert ag_lus[REF['beef_nat']] == 'Beef - natural land' and ag_lus[REF['sheep_nat']] == 'Sheep - natural land'
    assert len(nrm) == len(st) == len(ra) == len(lu) == NCELL

    # ---- projects: written once, read back (the rates use the written values)
    cells = pd.read_csv(upstream['hir_project_cells.csv'])
    t, ns = projects(src['cer_register'], pd.read_csv(upstream['hir_projects.csv']), cells, nrm, st, ra)
    p_proj = os.path.join(a.out, 'hir_projects_rates.csv')
    t.to_csv(p_proj, index=False, float_format='%.9g', lineterminator='\n')
    t = pd.read_csv(p_proj)
    g, sg, gg = region_rates(t, ns)
    g.to_csv(os.path.join(a.out, 'hir_region_rates.csv'), float_format='%.9g', lineterminator='\n')
    sg.to_csv(os.path.join(a.out, 'hir_state_rates.csv'), float_format='%.9g', lineterminator='\n')
    gg.to_csv(os.path.join(a.out, 'hir_region_rates_gross.csv'), float_format='%.9g', index_label='region', lineterminator='\n')

    # ---- layer
    rate = pd.Series(nrm).map(g.HIR_RATE).to_numpy()
    elig = np.isfinite(rate)
    iss = np.where(elig, rate, 0.0)
    gr = pd.Series(nrm).map(gg.HIR_RATE_gross).to_numpy()
    assert (np.isfinite(gr) == elig).all()
    G = np.where(elig, gr, 0.0)
    L = pd.DataFrame({'HIR_RATE': G.astype(np.float32), 'HIR_ELIGIBLE': elig.astype(np.float32), 'HIR_RATE_ISSUANCE': iss.astype(np.float32)})
    out_h5 = os.path.join(a.out, 'hir_rate_layer_gross.h5')
    L.to_hdf(out_h5, key='hir_rate_layer_gross', mode='w', format='table', complevel=9, complib='zlib')
    back = pd.read_hdf(out_h5)
    assert back.equals(L)
    ncs = [NC.write_like(a.out, 'hir_rate_layer_gross', L)]                         # NetCDF copy, jinzhu layout

    # ---- Beef / Sheep natural land (the --lumap map, the run's base year) in the eligible footprint, by state
    bn, sn = REF['beef_nat'], REF['sheep_nat']
    s = pd.DataFrame({'state_cell': st, 'ha': ra, 'elig': elig, 'bnat': lu == bn, 'snat': lu == sn})
    e = s[s.elig]
    bs = pd.DataFrame({'beef_nat_total_ha': s[s.bnat].groupby('state_cell').ha.sum(), 'beef_nat_eligible_ha': e[e.bnat].groupby('state_cell').ha.sum(),
                       'sheep_nat_total_ha': s[s.snat].groupby('state_cell').ha.sum(), 'sheep_nat_eligible_ha': e[e.snat].groupby('state_cell').ha.sum()}).fillna(0)
    bs.loc['Total'] = bs.sum()
    bs.to_csv(os.path.join(a.out, 'hir_eligible_beef_sheep_natural_by_state.csv'), float_format='%.9g', lineterminator='\n')
    r_ = t[t.in_rates]
    print(f'hir_layer: {len(t)} HIR projects, {int(t.in_rates.sum())} rated, {len(gg)} regions ({int(gg.pooled_to_state.sum())} pooled); '
          f'national gross {r_.gross_mean_FY20_24.sum() / r_.cea_ha.sum():.4f}, issuance {r_.iss_mean_FY20_24.sum() / r_.cea_ha.sum():.4f} '
          f't CO2e/ha/yr; {int(elig.sum())} eligible cells', f'{time.time() - t0:.0f}s', flush=True)

    # ---- provenance (last)
    me = open(os.path.abspath(__file__), 'rb').read()
    P = [['source', k, url, pin, 'sha256 of raw file', note] for k, (url, pin, note) in SOURCES.items()]
    if a.live_check:
        P.append(['live_check', 'cer_register', SOURCES['cer_register'][0], SOURCES['cer_register'][1], 'live file equal to pin',
                  time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())])
    P += [['inherited', f, '', in_sha[f], 'inherited 2010-build input (inherited_2010)', ''] for f in INHERITED]
    P += [['inherited' if a.base_year == 2010 else 'upstream', 'lumap.h5', '', lumap_sha,
           'inherited 2010-build input, the 2010 map, dataset content sha256 (pinned)' if a.base_year == 2010
           else 'the 2021 map (livestock_map.py), dataset content sha256 (pinned)', a.lumap.replace('\\', '/')]]
    P += [['upstream', f, '', in_sha[f], 'output of ag_mgt_caps.py', upstream[f].replace('\\', '/')] for f in upstream]
    P += [['reference', k, '', '', repr(v), 'reference table'] for k, v in REF.items()]
    P += [['script', 'input/Nick_build/scripts/hir_layer.py', '', sha(me), 'sha256 of script', '']]
    P += [['package', k, '', v, 'version', ''] for k, v in sorted(pkg_versions().items())]
    P += [['python', sys.version.split()[0], '', '', 'interpreter', '']]
    P.append(['output', 'hir_rate_layer_gross.h5', '', sha_file(out_h5), 'sha256 of file (HDF5 bytes vary between runs)', ''])
    for col in back.columns:
        P.append(['dataset', f'hir_rate_layer_gross.h5:/hir_rate_layer_gross {col}', '', content_sha(back[col].to_numpy()),
                  'sha256 of dtype, shape and array bytes', f'{back[col].dtype} {back[col].shape}'])
    P.append(['dataset', 'hir_rate_layer_gross.h5:/hir_rate_layer_gross values', '', content_sha(back.to_numpy()),
              'sha256 of dtype, shape and array bytes', f'{",".join(back.columns)} {back.to_numpy().dtype} {back.shape}'])
    P.append(['dataset', 'hir_rate_layer_gross.h5:/hir_rate_layer_gross index', '', content_sha(back.index.to_numpy()),
              'sha256 of dtype, shape and array bytes', 'cell order'])
    for f in ('hir_projects_rates.csv', 'hir_region_rates_gross.csv', 'hir_region_rates.csv', 'hir_state_rates.csv',
              'hir_eligible_beef_sheep_natural_by_state.csv'):
        P.append(['output', f, '', sha_file(os.path.join(a.out, f)), 'sha256', ''])
    P += [['netcdf', os.path.basename(p), '', h, 'sha256 of NetCDF content (nc_layout.content_sha)', 'checked equal to the .h5 arrays'] for p, h in ncs]
    write_csv(os.path.join(a.out, 'provenance.csv'), ['kind', 'item', 'url', 'sha256', 'note', 'source'], P)
    print('done', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
