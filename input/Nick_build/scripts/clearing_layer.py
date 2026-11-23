#!/usr/bin/env python
"""Build LUTO's land-clearing emissions layer (clearing_layer.h5): NVIS forest share x NIR primary-clearing emission per hectare.

Sources (public, downloaded at run time, sha256-checked against pinned values):
  - DCCEEW National Inventory Report 2024 activity table, LULUCF (Tables 6-14: national and state forest conversion,
    primary and re-clearing area and direct emissions by financial year), CC BY 4.0.
Inherited 2010-build inputs read from --input (inherited_2010/, copies of 2010-build files), sha256 recorded in provenance: bio_GBF3_NVIS_MVG.nc and
bio_GBF3_NVIS_MVG_groups.npy (NVIS major vegetation group percent cover per cell; inherited 2010-build input, origin not
recorded) and REGION_STATE_r.h5 (state of each cell). The output is therefore not built from public sources alone.

Method:
  1. Per state and nationally, pooled over FY2016-17 to FY2021-22: P = primary-clearing direct emissions / primary area
     (t CO2e/ha), X = all direct emissions / all cleared area. FY2022-23 and FY2023-24 copy FY2021-22 in the source and are
     excluded (checked, recorded in provenance). ACT has no primary clearing in the window and takes the NSW rates; cells of
     other territories take the national rate. VIC and TAS rates are kept and flagged.
  2. Per cell, shares of native NVIS cover (all groups less the three non-native ones): F_FOREST (groups meeting NIR's
     forest definition) and F_SPARSE (open woodlands); 0 where the cell has no native cover.
  3. CLEAR_DIRECT = F_FOREST x P_state (one-off direct emission at clearing), CLEAR_DIRECT_MIXED = F_FOREST x X_state.
  The pooled rate table is written once and read back before step 3.
Outputs in --out: clearing_layer.h5 (key clearing_layer, pandas table, zlib 9, float32 CLEAR_DIRECT, CLEAR_DIRECT_MIXED,
F_FOREST, F_SPARSE, int64 index 0..N-1); nir_rates_pooled_fy2017_2022.csv;
rates_used.csv; clearing_layer.nc (NetCDF copy in the jinzhu layout, one variable per column,
nc_layout.py); provenance.csv (last).

Run as a plain script from the repository root (no luto imports):

    python input/Nick_build/scripts/clearing_layer.py --input input/Nick_build/inherited_2010 --out input/Nick_build/clearing_layer --cache input/Nick_build/_cache
        [--offline] [--source NAME=PATH]

Dependencies: numpy, pandas, PyTables, openpyxl, xarray, netCDF4. No luto imports (nc_layout.py is a sibling module).
Memory: peak resident memory about 1.4 GB on the full grid (measured October 2026); run on a machine with at least 8
GB of RAM.
Output values and the CSV files are deterministic; HDF5 file bytes carry a group timestamp, so provenance records dataset
content hashes.
"""
import argparse
import csv
import hashlib
import os
import sys
import time
import urllib.request
from importlib import metadata

import numpy as np
import pandas as pd
import xarray as xr

import nc_layout as NC

# ----------------------------------------------------------------------------- sources and pins
SOURCES = {   # name: (url, sha256 of the raw file, note)
    'nir_lulucf': ('https://www.dcceew.gov.au/sites/default/files/documents/activity-table-1990-2024-lulucf.xlsx',
                   '2b3178a9a031c8fe8553a6768cabeaf2bc90dfa2877cb30a4ef781dc2dca4d83',
                   'DCCEEW NIR 2024 activity table, LULUCF, CC BY 4.0'),
}
INHERITED = ['bio_GBF3_NVIS_MVG.nc', 'bio_GBF3_NVIS_MVG_groups.npy', 'REGION_STATE_r.h5']
INHERITED_NOTE = {'bio_GBF3_NVIS_MVG.nc': 'inherited 2010-build input, origin not recorded',
                  'bio_GBF3_NVIS_MVG_groups.npy': 'inherited 2010-build input, origin not recorded',
                  'REGION_STATE_r.h5': 'inherited 2010-build input (inherited_2010)'}
NCELL = 6956407

# ============================================================================= reference table (hand-entered)
# Layout of the NIR activity tables (Tables 6-14): rows 8-42, first ten columns: FY label then these nine quantities.
NIR_COLS = ['prim_kha', 'prim_Mt', 'sec_kha', 'sec_Mt', 'decay_Mt', 'ident_regrowth_kha', 'sust_regrowth_kha',
            'regrowth_net_Mt', 'net_clearing_kha']
NIR_SHEETS = {'AUS': 'Table 6', 'QLD': 'Table 7', 'NSW': 'Table 8', 'VIC': 'Table 9', 'WA': 'Table 10', 'TAS': 'Table 11',
              'SA': 'Table 12', 'NT': 'Table 13', 'ACT': 'Table 14'}
# Pooling window FY2016-17 to FY2021-22 (FY ending 2017-2022); later years copy FY2021-22
FY_FIRST, FY_LAST = 2017, 2022
# ACT has no primary clearing in the window: it takes NSW rates; Other Territories (64 cells) take the national rate.
# VIC and TAS kept and flagged: each rests on
# about 5 kha primary clearing over six years.
ACT_RATES_FROM = 'NSW'
FLAGGED = {'VIC': 'kept and flagged: about 5 kha primary clearing in the window',
           'TAS': 'kept and flagged: about 5.5 kha primary clearing in the window'}
STATE_CODE = {'Queensland': 'QLD', 'New South Wales': 'NSW', 'Victoria': 'VIC', 'Western Australia': 'WA', 'Tasmania': 'TAS',
              'South Australia': 'SA', 'Northern Territory': 'NT', 'Australian Capital Territory': 'ACT'}   # else national
# NVIS major vegetation groups meeting NIR's forest definition (forests and woodlands, tropical eucalypt woodlands, mallee
# woodlands, closed shrublands, mangroves), the sparse (open woodland) groups and the non-native groups excluded from the
# native total
FOREST = ['Rainforests and Vine Thickets', 'Eucalypt Tall Open Forests', 'Eucalypt Open Forests', 'Eucalypt Low Open Forests',
          'Eucalypt Woodlands', 'Acacia Forests and Woodlands', 'Callitris Forests and Woodlands', 'Casuarina Forests and Woodlands',
          'Melaleuca Forests and Woodlands', 'Other Forests and Woodlands', 'Tropical Eucalypt Woodlands/Grasslands',
          'Low Closed Forests and Tall Closed Shrublands', 'Mangroves', 'Mallee Woodlands and Shrublands', 'Unclassified Forest']
SPARSE = ['Eucalypt Open Woodlands', 'Acacia Open Woodlands', 'Other Open Woodlands',
          'Mallee Open Woodlands and Sparse Mallee Shrublands']
NONNATIVE = ['Inland aquatic - freshwater, salt lakes, lagoons', 'Naturally bare - sand, rock, claypan, mudflat', 'Sea and estuaries']
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
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (luto-clearing-layer/1.0)'})
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


# ----------------------------------------------------------------------------- method
def nir_activity(xlsx):
    """Tables 6-14 in long form. The NT FY1998-99 row is garbled text in the source (NaN, outside
    the window)."""
    rows = []
    for k, s in NIR_SHEETS.items():
        df = pd.read_excel(xlsx, sheet_name=s, header=None)
        d = df.iloc[7:42, :10].copy()
        d.columns = ['fy'] + NIR_COLS
        d = d[d.fy.astype(str).str.match(r'^\d{4}-\d{2}$')]
        for c in NIR_COLS:
            d[c] = pd.to_numeric(d[c], errors='coerce')
        d['fy_end'] = d.fy.str[:4].astype(int) + 1
        d['region'] = k
        rows.append(d)
    return pd.concat(rows)


def carry_forward(A):
    """Regions whose FY2022-23 and FY2023-24 non-decay values equal FY2021-22 (the reason the window ends at FY2021-22)."""
    keep = [c for c in NIR_COLS if c != 'decay_Mt']
    out = {}
    for k in NIR_SHEETS:
        d = A[A.region == k].set_index('fy')
        out[k] = all(np.allclose(d.loc[fy, keep].values.astype(float), d.loc['2021-22', keep].values.astype(float))
                     for fy in ('2022-23', '2023-24'))
    return out


def pooled_rates(A):
    W = A[(A.fy_end >= FY_FIRST) & (A.fy_end <= FY_LAST)]

    def rates(g):
        return pd.Series(dict(prim_kha=g.prim_kha.sum(), prim_Mt=g.prim_Mt.sum(), sec_kha=g.sec_kha.sum(), sec_Mt=g.sec_Mt.sum(),
                              P=g.prim_Mt.sum() / g.prim_kha.sum() * 1000 if g.prim_kha.sum() > 0 else np.nan,
                              S=g.sec_Mt.sum() / g.sec_kha.sum() * 1000 if g.sec_kha.sum() > 0 else np.nan,
                              X=(g.prim_Mt.sum() + g.sec_Mt.sum()) / (g.prim_kha.sum() + g.sec_kha.sum()) * 1000,
                              sec_share_area=g.sec_kha.sum() / (g.prim_kha.sum() + g.sec_kha.sum()),
                              regrow_kha_yr=g.sust_regrowth_kha.sum(), regrow_Mt=g.regrowth_net_Mt.sum(),
                              REGROW_t_ha_yr=g.regrowth_net_Mt.sum() / g.sust_regrowth_kha.sum() * 1000))
    return W.groupby('region').apply(rates).reindex(list(NIR_SHEETS))


def shares(nc, groups_npy):
    """Forest and sparse shares of native NVIS cover per cell."""
    groups = [str(g) for g in np.load(groups_npy, allow_pickle=True)]
    ds = xr.open_dataset(nc)
    assert list(ds.group.values) == groups
    missing = [g for g in FOREST + SPARSE if g not in groups]
    V = ds['data'].values            # (groups, cells) uint8 percent
    ds.close()
    gi = {g: i for i, g in enumerate(groups)}
    nat_idx = [i for g, i in gi.items() if g not in NONNATIVE]
    tot_native = V[nat_idx].sum(0, dtype=np.int32)
    fo = V[[gi[g] for g in FOREST if g in gi]].sum(0, dtype=np.int32)
    sp = V[[gi[g] for g in SPARSE if g in gi]].sum(0, dtype=np.int32)
    ok = tot_native > 0
    fF = np.where(ok, fo / np.maximum(tot_native, 1), 0.).astype(np.float32)
    fS = np.where(ok, sp / np.maximum(tot_native, 1), 0.).astype(np.float32)
    return fF, fS, missing, int((~ok).sum())


# ----------------------------------------------------------------------------- provenance
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--input', required=True, help='folder of the inherited 2010-build files listed in INHERITED (input/Nick_build/inherited_2010)')
    ap.add_argument('--out', required=True, help='output folder')
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
    src = {k: fetch(k, a.cache, a.offline, overrides) for k in SOURCES}
    inp = {f: os.path.join(a.input, f) for f in INHERITED}
    in_sha = {f: sha_file(p) for f, p in inp.items()}

    # ---- NIR rates: written once, read back (the layer uses the written values)
    A = nir_activity(src['nir_lulucf'])
    cf = carry_forward(A)
    p_rates = os.path.join(a.out, 'nir_rates_pooled_fy2017_2022.csv')
    pooled_rates(A).to_csv(p_rates, lineterminator='\n')
    R = pd.read_csv(p_rates, index_col=0)
    P, X = R.P.copy(), R.X.copy()
    P['ACT'] = P[ACT_RATES_FROM]
    X['ACT'] = X[ACT_RATES_FROM]

    # ---- layer
    fF, fS, missing, no_native = shares(inp['bio_GBF3_NVIS_MVG.nc'], inp['bio_GBF3_NVIS_MVG_groups.npy'])
    st = pd.read_hdf(inp['REGION_STATE_r.h5'])['STE_NAME11'].astype(str).to_numpy()
    assert len(st) == len(fF) == NCELL
    code = np.array([STATE_CODE.get(s, 'AUS') for s in st])
    Pcell = pd.Series(code).map(P).to_numpy()
    Xcell = pd.Series(code).map(X).to_numpy()
    df = pd.DataFrame({'CLEAR_DIRECT': (fF * Pcell), 'CLEAR_DIRECT_MIXED': (fF * Xcell), 'F_FOREST': fF, 'F_SPARSE': fS}).astype('float32')
    out_h5 = os.path.join(a.out, 'clearing_layer.h5')
    df.to_hdf(out_h5, key='clearing_layer', mode='w', format='table', complevel=9, complib='zlib')
    back = pd.read_hdf(out_h5)
    assert back.equals(df)
    ncs = [NC.write_like(a.out, 'clearing_layer', df)]                              # NetCDF copy, jinzhu layout
    note = ['national' if r == 'AUS' else (f'{ACT_RATES_FROM} rate (no primary clearing in window)' if r == 'ACT'
            else 'own state pooled FY2017-FY2022') for r in P.index]
    pd.DataFrame({'region': list(P.index), 'P_used': P.values, 'X_used': X.values, 'note': note}).to_csv(
        os.path.join(a.out, 'rates_used.csv'), index=False, lineterminator='\n')
    print('clearing_layer: P (t CO2e/ha)', {k: round(v, 1) for k, v in P.items()}, f'carry-forward {cf}',
          f'other-territory cells {int((code == "AUS").sum())}, no-native cells {no_native}', f'{time.time() - t0:.0f}s', flush=True)

    # ---- provenance (last)
    me = open(os.path.abspath(__file__), 'rb').read()
    P_ = [['source', k, url, pin, 'sha256 of raw file', note_] for k, (url, pin, note_) in SOURCES.items()]
    P_ += [['inherited', f, '', in_sha[f], INHERITED_NOTE[f], ''] for f in INHERITED]
    P_ += [['check', 'carry_forward', '', '', 'FY2022-23 and FY2023-24 equal FY2021-22 (non-decay columns)',
            ';'.join(f'{k}={v}' for k, v in cf.items())],
           ['check', 'forest_sparse_groups_missing', '', '', 'FOREST/SPARSE names not in the NVIS file', ';'.join(missing) or 'none']]
    P_ += [['reference', k, '', '', repr(v), 'reference table'] for k, v in
           dict(FY_FIRST=FY_FIRST, FY_LAST=FY_LAST, ACT_RATES_FROM=ACT_RATES_FROM, FLAGGED=FLAGGED, STATE_CODE=STATE_CODE,
                FOREST=FOREST, SPARSE=SPARSE, NONNATIVE=NONNATIVE, NIR_SHEETS=NIR_SHEETS, NIR_COLS=NIR_COLS).items()]
    P_ += [['script', 'input/Nick_build/scripts/clearing_layer.py', '', sha(me), 'sha256 of script', '']]
    P_ += [['package', k, '', v, 'version', ''] for k, v in sorted(pkg_versions().items())]
    P_ += [['python', sys.version.split()[0], '', '', 'interpreter', '']]
    P_.append(['output', 'clearing_layer.h5', '', sha_file(out_h5), 'sha256 of file (HDF5 bytes vary between runs)', ''])
    for col in back.columns:
        P_.append(['dataset', f'clearing_layer.h5:/clearing_layer {col}', '', content_sha(back[col].to_numpy()),
                   'sha256 of dtype, shape and array bytes', f'{back[col].dtype} {back[col].shape}'])
    P_.append(['dataset', 'clearing_layer.h5:/clearing_layer values', '', content_sha(back.to_numpy()),
               'sha256 of dtype, shape and array bytes', f'{",".join(back.columns)} {back.to_numpy().dtype} {back.shape}'])
    P_.append(['dataset', 'clearing_layer.h5:/clearing_layer index', '', content_sha(back.index.to_numpy()),
               'sha256 of dtype, shape and array bytes', 'cell order'])
    for f in ('nir_rates_pooled_fy2017_2022.csv', 'rates_used.csv'):
        P_.append(['output', f, '', sha_file(os.path.join(a.out, f)), 'sha256', ''])
    P_ += [['netcdf', os.path.basename(p), '', h, 'sha256 of NetCDF content (nc_layout.content_sha)', 'checked equal to the .h5 arrays'] for p, h in ncs]
    write_csv(os.path.join(a.out, 'provenance.csv'), ['kind', 'item', 'url', 'sha256', 'note', 'source'], P_)
    print('done', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
