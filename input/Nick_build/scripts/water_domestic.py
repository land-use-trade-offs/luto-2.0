#!/usr/bin/env python
"""Build LUTO's 2021-base domestic and industrial water use by water region (Water_Use_Domestic.csv).

Sources (public, downloaded at run time, sha256-checked against pinned values):
  - ABS Water Account, Australia, 2020-21 (cat. 4610.0, released 2022-10-20), 46100DO014: physical water supply and use by
    industry, water type, Australia and state/territory, 2014-15 to 2020-21 (ML).
  - The same release, 46100DO001 Table 1.4: water consumption, Australia (the published national check).
Inherited 2010-build inputs read from --input (inherited_2010/, copies of 2010-build files), sha256 recorded in provenance: Water_Use_Domestic.csv (the
2010-11 file this replaces: ABS Water Account 2010-11, households plus ANZSIC divisions B to S, by drainage division and
river region; sha256 pinned), draindiv_id.h5, rivreg_id.h5, draindiv_lut.h5, rivreg_lut.h5, REGION_STATE_r.h5,
real_area.h5. The output is therefore not built from public sources alone.

Method:
  1. 2020-21 use by state, households plus non-agricultural industries, not crediting the water industry's effluent
     returns: Q = total consumption - agriculture, forestry and fishing (AFF) consumption + the water supply, sewerage
     and drainage industry's return flows, where
       - total consumption = self-extracted use - return flows (the identity of the published consumption, Table 1.4);
       - AFF consumption = AFF use (self-extracted, distributed, reuse) - AFF supply (distributed, reuse, return flows);
       - a suppressed (np) AFF cell is the residual of the published state total less the other published rows when it is
         the only suppressed cell (exact); where other rows are suppressed too, the AFF of those states jointly is the
         Australia AFF less the other states', split by each state's published AFF self-extracted use (estimated).
     Q covers households and ANZSIC B to S: mining, manufacturing, electricity and gas supply, water supply, sewerage
     and drainage services, waste collection, treatment and disposal services, other industries.
  2. Each river region's 2010-11 value is split across states by the real area of its LUTO cells in each state
     (REGION_STATE_r.h5); per scaling group (each state, except NSW and the ACT as one group, as NIR reports them and
     their river regions are shared), the scale factor = Q / the group's 2010-11 total; each river region's 2021 value =
     the sum over states of its state parts x the group's factor. Each drainage division is the sum of the river regions
     lying mostly in it (by cell area), as the 2010-11 file is built (it nests so but for 3.427 ML between divisions 11
     and 13). Both region types sum to Q by scaling group. Other Territories (no ABS 2020-21 figure) keep their 2010-11
     part (factor 1).
  The spatial pattern is the 2010-11 file's. That file is 0.9838 of the ABS 2010-11 households plus B to S total
  (5,889,627 against 5,986,324 ML); the 96.7 GL gap is unattributed (its apportionment to regions is not recorded).
Outputs in --out: Water_Use_Domestic.csv (layout, row order and line endings of the 2010-11 file);
water_domestic_state_derivation.csv (every term of Q by state, with its basis); water_domestic_scale_by_state.csv (2010-11
total, Q and factor by state); water_domestic_by_region.csv (2010-11 and 2021 values by region);
provenance.csv (last).

Run as a plain script from the repository root (no luto imports):

    python input/Nick_build/scripts/water_domestic.py --input input/Nick_build/inherited_2010 --out input/Nick_build/water_domestic --cache input/Nick_build/_cache
        [--offline] [--live-check] [--source NAME=PATH]

Dependencies: numpy, pandas, PyTables, openpyxl. No luto imports. Peak memory about 1 GB. Output is deterministic.
"""
import argparse
import csv
import hashlib
import io
import os
import sys
import time
import urllib.request
from importlib import metadata

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- sources and pins
ABS = 'https://www.abs.gov.au/statistics/environment/environmental-accounts/water-account-australia/2020-21/'
SOURCES = {   # name: (url, sha256 of the raw file, note)
    'abs_4610_do014': (ABS + '46100DO014_202021.csv', '008c1ed73708d9d639d6595b56688ee4780ecedbe4741c24fa2445dcc688fd1e',
                       'ABS 4610.0 2020-21, physical water supply and use by industry and water type, Australia and states (ML)'),
    'abs_4610_do001': (ABS + '46100DO001_202021.xlsx', 'e2fa9f6d8352ebcc7c80df57ac0f0d9ef84c6faeb48c7b266545c38ec0f52f91',
                       'ABS 4610.0 2020-21, summary indicators; Table 1.4 water consumption, Australia'),
}
INHERITED = ['Water_Use_Domestic.csv', 'draindiv_id.h5', 'rivreg_id.h5', 'draindiv_lut.h5', 'rivreg_lut.h5', 'REGION_STATE_r.h5',
             'real_area.h5']
OLD_SHA = 'ea54512e94b02ddf63a502f77d8473cf0cdbb20a0b93b9eb4ddfa2339654c42d'   # the 2010-11 Water_Use_Domestic.csv
FY = '2020-21'
STATES = ['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT']
SCALE_GROUP = {'NSW': 'NSW+ACT', 'ACT': 'NSW+ACT'}   # states scaled together; the others are their own group
GEO = {'Australia': 'AUS', 'New South Wales': 'NSW', 'Victoria': 'VIC', 'Queensland': 'QLD', 'South Australia': 'SA',
       'Western Australia': 'WA', 'Tasmania': 'TAS', 'Northern Territory': 'NT', 'Australian Capital Territory': 'ACT'}
STE11 = {'New South Wales': 'NSW', 'Victoria': 'VIC', 'Queensland': 'QLD', 'South Australia': 'SA', 'Western Australia': 'WA',
         'Tasmania': 'TAS', 'Northern Territory': 'NT', 'Australian Capital Territory': 'ACT', 'Other Territories': 'OT'}
REGIONS = {'Drainage Division': ('draindiv_id.h5', 'draindiv_lut.h5', 'HR_DRAINDIV_ID', 'HR_DRAINDIV_NAME'),
           'River Region': ('rivreg_id.h5', 'rivreg_lut.h5', 'HR_RIVREG_ID', 'HR_RIVREG_NAME')}
AFF = 'Total agriculture, forestry and fishing (c)'
WS = 'Water supply, sewerage and drainage services'
# rows that with AFF and households make up the state total of a product (division totals; Australia names manufacturing
# 'Total manufacturing', the states 'Manufacturing')
OTHER_ROWS = ['Total mining', 'Manufacturing', 'Total electricity, gas, water and waste services', 'Other industries', 'Households']


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
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (luto-water-domestic/1.0)'})
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
    for n in names:
        url, pin, _ = SOURCES[n]
        got = sha(download(url))
        if got != pin:
            raise SystemExit(f'{n}: live file {url} has sha256 {got}, pinned {pin}; the source has changed')


# ----------------------------------------------------------------------------- step 1: 2020-21 use by state
def supply_use(path):
    """DO014 as a lookup (geo, industry, side, product) -> (value ML, raw cell); '-' is 0; np and na kept as flags."""
    d = pd.read_csv(path, skiprows=4, encoding='latin-1', dtype=str)
    d.columns = [c.strip() for c in d.columns]
    d = d[d['Geography Code'].notna() & d['Unit'].eq('ML')].copy()
    d['geo'] = d['Geography Description'].map(GEO)
    assert d.geo.notna().all()
    d['raw'] = d[FY].str.strip()
    d['v'] = pd.to_numeric(d.raw.str.replace(',', '', regex=False).replace({'-': '0'}), errors='coerce')
    assert d.v.notna().eq(~d.raw.isin(['np', 'na'])).all(), 'unexpected cell text in DO014'
    d['Industry'] = d.Industry.replace({'Total manufacturing': 'Manufacturing'})
    idx = d.set_index(['geo', 'Industry', 'Supply/Use', 'Product'])
    assert idx.index.is_unique
    return idx


def state_quantities(T):
    def cell(g, ind, su, pr):
        r = T.loc[(g, ind, su, pr)]
        return (0.0 if pd.isna(r.v) else float(r.v)), r.raw
    rows = []
    for g in ['AUS'] + STATES:
        se, _ = cell(g, 'Total', 'Use', 'Self-extracted')
        rf, _ = cell(g, 'Total', 'Supply', 'Return flows (a)')
        u = {p: cell(g, AFF, 'Use', p) for p in ('Self-extracted', 'Distributed', 'Reuse')}
        s = {p: cell(g, AFF, 'Supply', p) for p in ('Distributed', 'Reuse', 'Return flows (a)')}
        assert all(r != 'np' for _, r in s.values()) and all(r != 'np' for p, (_, r) in u.items() if p != 'Distributed'), g
        basis, aff_d = 'published', u['Distributed'][0]
        if u['Distributed'][1] == 'np':
            oth = {o: cell(g, o, 'Use', 'Distributed') for o in OTHER_ROWS}
            sup = [o for o, (_, r) in oth.items() if r == 'np']
            aff_d = cell(g, 'Total', 'Use', 'Distributed')[0] - sum(v for v, _ in oth.values())
            basis = 'residual of the published state total (exact)' if not sup else 'estimated: pooled with ' + ', '.join(sup)
        aff = u['Self-extracted'][0] + aff_d + u['Reuse'][0] - sum(v for v, _ in s.values())
        ws_rf, wr = cell(g, WS, 'Supply', 'Return flows (a)')
        assert wr not in ('np', 'na'), g
        rows.append(dict(state=g, total_self_extracted_use=se, total_return_flows=rf, total_consumption=se - rf,
                         aff_use_self_extracted=u['Self-extracted'][0], aff_use_distributed=aff_d, aff_use_reuse=u['Reuse'][0],
                         aff_supply=sum(v for v, _ in s.values()), aff_consumption=aff, aff_basis=basis,
                         water_industry_return_flows=ws_rf))
    D = pd.DataFrame(rows).set_index('state')
    est = D.index[D.aff_basis.str.startswith('estimated')]
    if len(est):                 # their AFF jointly = Australia AFF - the other states', split by AFF self-extracted use
        joint = D.loc['AUS', 'aff_consumption'] - D.loc[[s for s in STATES if s not in est], 'aff_consumption'].sum()
        w = D.loc[est, 'aff_use_self_extracted']
        D.loc[est, 'aff_consumption'] = joint * w / w.sum()
        D.loc[est, 'aff_basis'] = (D.loc[est, 'aff_basis'] + f'; joint AFF {joint:.2f} ML (Australia less the other states) '
                                   'split by AFF self-extracted use')
    D['households_and_non_agricultural'] = D.total_consumption - D.aff_consumption + D.water_industry_return_flows
    return D


# ----------------------------------------------------------------------------- step 2: scale the 2010-11 regions
def scale_regions(old, inp, Q):
    """River regions scaled by state; drainage divisions as the sums of their river regions, as the 2010-11 file is built
    (each of its drainage divisions equals the sum of the river regions lying mostly in it, but for 3.427 ML between
    divisions 11 and 13: two river regions of exactly that value, one of them assigned off its majority division)."""
    ra = pd.read_hdf(inp['real_area.h5']).to_numpy().astype(np.float64)
    ste = pd.read_hdf(inp['REGION_STATE_r.h5'])['STE_NAME11'].map(STE11).to_numpy()
    assert not pd.isna(ste).any()
    rr = pd.read_hdf(inp['rivreg_id.h5']).to_numpy()
    dd = pd.read_hdf(inp['draindiv_id.h5']).to_numpy()
    sh = pd.DataFrame(dict(r=rr, s=ste, a=ra)).groupby(['r', 's']).a.sum().unstack(fill_value=0.0)
    sh = sh.div(sh.sum(axis=1), axis=0)
    rr_dd = pd.DataFrame(dict(r=rr, d=dd, a=ra)).groupby(['r', 'd']).a.sum().unstack(fill_value=0.0).idxmax(axis=1)
    o = {t: old[old.REGION_TYPE == t].set_index('REGION_ID').DOMESTIC_INDUSTRIAL_WATER_USE_ML for t in REGIONS}
    assert set(o['River Region'].index) <= set(sh.index), 'river regions with no LUTO cell'
    nest = o['River Region'].groupby(rr_dd.reindex(o['River Region'].index)).sum() - o['Drainage Division']
    assert nest.abs().max() < 3.43, nest                               # the 2010-11 file's own nesting
    parts = sh.reindex(o['River Region'].index).mul(o['River Region'], axis=0)       # river region x state, 2010-11
    old_s = parts.sum()
    grp = {s: SCALE_GROUP.get(s, s) for s in parts.columns}           # NSW and ACT scale as one group
    old_g = old_s.groupby(grp).sum()
    Q_g = Q.groupby(lambda s: SCALE_GROUP.get(s, s)).sum()
    assert (old_g.drop('OT', errors='ignore') > 0).all()
    k_g = pd.Series({g: (Q_g[g] / old_g[g] if g in Q_g else 1.0) for g in old_g.index})
    k = pd.Series({s: k_g[grp[s]] for s in parts.columns})
    new_parts = parts.mul(k, axis=1)
    v = {'River Region': new_parts.sum(axis=1)}
    v['Drainage Division'] = v['River Region'].groupby(rr_dd.reindex(v['River Region'].index)).sum().reindex(o['Drainage Division'].index)
    dd_parts = new_parts.groupby(rr_dd.reindex(new_parts.index)).sum()
    scale = [dict(state=s, scale_group=grp[s], old_2010_11_ml=old_s[s], new_2020_21_ml=new_parts[s].sum(), factor=k[s],
                  group_2020_21_ml=Q_g.get(grp[s], np.nan), new_2020_21_ml_drainage_divisions=dd_parts[s].sum(),
                  basis='ABS 2020-21 households and non-agricultural use' if s in Q else 'no ABS 2020-21 figure: kept')
             for s in parts.columns]
    new, byreg = [], []
    for rtype, (_, flut, cid, cname) in REGIONS.items():
        names = pd.read_hdf(inp[flut]).set_index(cid)[cname]
        new.append(pd.DataFrame({'REGION_TYPE': rtype, 'REGION_ID': v[rtype].index, 'DOMESTIC_INDUSTRIAL_WATER_USE_ML': v[rtype].to_numpy()}))
        for r in o[rtype].index:
            byreg.append(dict(region_type=rtype, region_id=r, region_name=names.get(r, ''), old_2010_11_ml=o[rtype][r],
                              new_2020_21_ml=v[rtype][r], ratio=v[rtype][r] / o[rtype][r] if o[rtype][r] else np.nan))
    return pd.concat(new, ignore_index=True), pd.DataFrame(scale), pd.DataFrame(byreg), nest


def pkg_versions():
    out = {}
    for p in ('numpy', 'pandas', 'tables', 'openpyxl'):
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--input', required=True, help='folder of the inherited 2010-build files listed in INHERITED (input/Nick_build/inherited_2010)')
    ap.add_argument('--out', required=True, help='output folder (e.g. input/Nick_build/water_domestic)')
    ap.add_argument('--cache', required=True, help='shared raw-download cache (content-addressed by sha256)')
    ap.add_argument('--offline', action='store_true', help='fail instead of downloading on a cache miss')
    ap.add_argument('--live-check', action='store_true', help='also fetch the live files and stop if any differs from its pin')
    ap.add_argument('--source', action='append', default=[], metavar='NAME=PATH',
                    help='a local copy of a pinned source (must match its pin); NAME in ' + ', '.join(SOURCES))
    a = ap.parse_args()
    t0 = time.time()
    overrides = dict(s.split('=', 1) for s in a.source)
    assert set(overrides) <= set(SOURCES), set(overrides) - set(SOURCES)
    inp = {f: os.path.join(a.input, f) for f in INHERITED}
    in_sha = {f: sha_file(p) for f, p in inp.items()}
    if in_sha['Water_Use_Domestic.csv'] != OLD_SHA:
        raise SystemExit(f"{inp['Water_Use_Domestic.csv']} has sha256 {in_sha['Water_Use_Domestic.csv']}, not the 2010-11 file "
                         f'({OLD_SHA}); refusing to run')
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    src = {k: fetch(k, a.cache, a.offline, overrides) for k in SOURCES}
    if a.live_check:
        live_check(list(SOURCES))

    # ---- step 1
    D = state_quantities(supply_use(src['abs_4610_do014']))
    t14 = pd.read_excel(src['abs_4610_do001'], sheet_name='Table 1.4', header=None)
    hdr = t14.index[t14.eq('Region').any(axis=1)][0]
    t14 = t14.iloc[hdr + 1:].set_axis([str(c).strip() for c in t14.iloc[hdr]], axis=1)
    pub = float(t14[(t14.Region == 'Australia') & (t14.Measure == 'Water Consumption')][FY].iloc[0])
    assert abs(D.loc['AUS', 'total_consumption'] - pub) < 1.0, (D.loc['AUS', 'total_consumption'], pub)
    assert abs(D.loc[STATES, 'aff_consumption'].sum() - D.loc['AUS', 'aff_consumption']) < 1.0
    Q = D.loc[STATES, 'households_and_non_agricultural']

    # ---- step 2
    old = pd.read_csv(inp['Water_Use_Domestic.csv'])
    new, S, R, nest = scale_regions(old, inp, Q)
    out = old[['REGION_TYPE', 'REGION_ID']].merge(new, on=['REGION_TYPE', 'REGION_ID'], how='left')
    assert list(out.columns) == list(old.columns) and len(out) == len(old) and out.notna().all().all()
    s = S.set_index('state')
    g = s.loc[STATES].groupby('scale_group')
    assert np.allclose(g.new_2020_21_ml.sum(), g.group_2020_21_ml.first(), rtol=1e-12)
    assert np.allclose(g.new_2020_21_ml_drainage_divisions.sum(), g.group_2020_21_ml.first(), rtol=1e-12)
    for rtype in REGIONS:
        assert abs(out[out.REGION_TYPE == rtype].DOMESTIC_INDUSTRIAL_WATER_USE_ML.sum() - s.new_2020_21_ml.sum()) < 1e-3, rtype
    eol = '\r\n' if b'\r\n' in open(inp['Water_Use_Domestic.csv'], 'rb').read() else '\n'   # line ends of the 2010-11 file
    out.to_csv(os.path.join(a.out, 'Water_Use_Domestic.csv'), index=False, lineterminator=eol)
    D.reset_index().to_csv(os.path.join(a.out, 'water_domestic_state_derivation.csv'), index=False, lineterminator='\n', float_format='%.2f')
    S.drop(columns='new_2020_21_ml_drainage_divisions').to_csv(os.path.join(a.out, 'water_domestic_scale_by_state.csv'),
                                                                index=False, lineterminator='\n', float_format='%.6f')
    R.to_csv(os.path.join(a.out, 'water_domestic_by_region.csv'), index=False, lineterminator='\n', float_format='%.6f')
    print('water_domestic: 2020-21 households and non-agricultural use', f"{Q.sum():,.0f} ML",
          f"(Australia {D.loc['AUS', 'households_and_non_agricultural']:,.0f}); 2010-11 file",
          f"{old[old.REGION_TYPE == 'Drainage Division'].DOMESTIC_INDUSTRIAL_WATER_USE_ML.sum():,.0f} ML", f'{time.time() - t0:.0f}s')

    # ---- provenance (last)
    me = open(os.path.abspath(__file__), 'rb').read()
    P = [['source', k, url, pin, 'sha256 of raw file', note] for k, (url, pin, note) in SOURCES.items()]
    if a.live_check:
        P += [['live_check', k, SOURCES[k][0], SOURCES[k][1], 'live file equal to pin', time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())]
              for k in SOURCES]
    P += [['inherited', f, '', in_sha[f], 'inherited 2010-build input (inherited_2010)',
           'ABS Water Account 2010-11, households plus ANZSIC B to S, by region (pinned); known limitation: 0.9838 of the '
           'ABS 2010-11 total, 96.7 GL unattributed' if f == 'Water_Use_Domestic.csv' else ''] for f in INHERITED]
    P += [['check', 'Australia consumption 2020-21', '', '', f'derived {D.loc["AUS", "total_consumption"]:.2f}, published {pub:.2f} ML',
           'DO001 Table 1.4']]
    P += [['check', f'2010-11 file nesting, drainage division {r}', '', '', f'sum of river regions - division value {v:.6f} ML',
           'river regions assigned to the division holding most of their area'] for r, v in nest.items() if abs(v) > 1e-6]
    P += [['estimated', f'AFF consumption {s}', '', '', f'{D.loc[s, "aff_consumption"]:.2f} ML', D.loc[s, 'aff_basis']]
          for s in STATES if D.loc[s, 'aff_basis'] != 'published']
    P += [['script', 'input/Nick_build/scripts/water_domestic.py', '', sha(me), 'sha256 of script', '']]
    P += [['package', k, '', v, 'version', ''] for k, v in sorted(pkg_versions().items())]
    P += [['python', sys.version.split()[0], '', '', 'interpreter', '']]
    for f in ('Water_Use_Domestic.csv', 'water_domestic_state_derivation.csv', 'water_domestic_scale_by_state.csv',
              'water_domestic_by_region.csv'):
        P.append(['output', f, '', sha_file(os.path.join(a.out, f)), 'sha256', ''])
    write_csv(os.path.join(a.out, 'provenance.csv'), ['kind', 'item', 'url', 'sha256', 'note', 'source'], P)
    print('done', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
