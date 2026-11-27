#!/usr/bin/env python
"""Build LUTO's pasture productivity layer (pasture_kg_dm_ha.h5) from AussieGRASS total pasture growth, FY2016-FY2025.

Sources (public, downloaded at run time, checked against pinned values):
  - Queensland Government LongPaddock AussieGRASS national grids, "12months.growth.tot" (total pasture growth, kg DM/ha,
    July to June), issues 2016-06 to 2025-06, ERDAS Imagine in zip (CC BY 4.0).
    Pinned on the decoded grid, not the file bytes (LongPaddock grids only): sha256 of the array's dtype,
    shape, transform, CRS and nodata attribute (none is set; -2 lake and 0 are hashed as stored values) and of its pixel
    bytes as read by GDAL. LongPaddock rebuilds recent issues with new trailing metadata and identical values (issue
    2025-06, rebuilt 2026-10-08). The raw-file sha256 is recorded but not enforced; any change in a pixel value, the
    layout or the dtype stops the run and sets the download aside.
Inherited 2010-build input read from --input (inherited_2010/, copies of 2010-build files), sha256 recorded in provenance: NLUM_2010-11_mask.tif (the cell
grid). The output is therefore not built from public sources alone.

Method:
  1. Mean of the ten annual grids (0.05 degree) per pixel.
  2. Invalid pixel: lake (-2 in any year) or 0 in every year, on land or not; no land mask.
  3. Per LUTO cell centre, bilinear sampling with corner weights snapped to 0 or 1 within 1e-9, using only valid corners with
     weight > 0, renormalised; where every weighted corner is invalid, the nearest valid pixel centre (Euclidean, degrees),
     exact-distance ties (within 1e-9 degree) broken by the lowest pixel index (row-major), whatever the scipy build.
  4. Rounded to whole kg DM/ha, int16.
Outputs in --out: pasture_kg_dm_ha.h5 (pandas Series PASTURE_KG_DM_HA, int16, index 0..N-1, key pasture_kg_dm_ha, table
format, zlib 9, the layout of the 2010-base file of the same name); pasture_kg_dm_ha.nc (variable PASTURE_KG_DM_HA over
`cell`, the jinzhu branch's NetCDF layout, nc_layout.py); provenance.csv (last), which also
lists, for every nearest-filled cell with a tie, the tied pixels and the one chosen.

Run as a plain script from the repository root (no luto imports):

    python input/Nick_build/scripts/pasture_productivity.py --input input/Nick_build/inherited_2010 --out input/Nick_build/pasture_productivity
        --cache input/Nick_build/_cache [--offline] [--source NAME=PATH]

Dependencies: numpy, pandas, PyTables, rasterio, scipy, xarray, netCDF4. No luto imports (nc_layout.py is a sibling module).
Memory: peak resident memory about 1.2 GB on the full grid (measured October 2026); run on a machine with at least 8
GB of RAM.
Output values are deterministic. HDF5 files carry an object modification time on the group pandas writes, so the file
bytes differ between runs; provenance records the sha256 of the dataset content (dtype, shape and array bytes) and index.
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
import rasterio
from affine import Affine

import nc_layout as NC

# ----------------------------------------------------------------------------- sources and pins
LP = 'https://data.longpaddock.qld.gov.au/AussieGRASS2/AUS/'
YEARS = list(range(2016, 2026))                       # issue YYYY06 = July (YYYY-1) to June YYYY
# year: (sha256 of the decoded grid (enforced), sha256 of the raw zip retrieved 2026-09-28 (recorded, not enforced))
PINS = {2016: ('8f8650588b205f141eb1ce6544e68a05f7372c0946f19778921f80e9e49df438', '9d975d145ac3d03d8a1f832154640265b40bdedbf5b4c8828f8d4165b337ec9e'),
        2017: ('f45c6b56332e9c0794faac543932cbca44d55ee1812e467c8885f944e0b0688e', 'a89926b7e9e2b0dfcc322de27c2531c2dd2049ac06bb656a774c22dea69299b2'),
        2018: ('565cfaba9d47bb06cd516e78507a60c20a941c392c6c2509ae46d7c47fc2fcd8', '9d714ce53801bb5b5052fb048341a6acd3cd849659c46e958b190e61ac58f37c'),
        2019: ('44e71897832ef343bc5c38815035f3bd34bb49573eb9a4f759c96d7b20b7f8bf', '9b667357da14249688421860d609c1146b735cf81638610e85a7dacc567c2705'),
        2020: ('2db29d3d7785f962c417c518e10c99397da0f52b6c97b2488112084d6ca11780', '59a4c70df4aed95490d3ef178cb282dd59b22cdfb42a33353aeb59edbef8eda3'),
        2021: ('ab40ab92b6e436a44f8082f0717bb6c500030708587f0174c685ab9a92effe63', '3596e2d2b76e2fd7c1362f5318916d12ea2f318d1971480c9e65e709e17975ac'),
        2022: ('b77558e81e69ff6fbd25e4fa373bff4aed7b4ce2dad77394ee1a85174aa78ecb', '545ef3f19b646397a964842ef07142ba5ddd5954e54b3521a18b953d0b78f4de'),
        2023: ('4b615bcae48e86cc4d71f722b8e919300c6ba3f9bbe5ee663dc0f00b76966ea2', '7a6eb54ffc851fc5b25cd964c5a36c169f8b8c6a00f9e646ac6357dc8db9d486'),
        2024: ('029b6a184a1bc967fb778674e2cb3809d239df0670c5e29c3cc72d879aedc383', 'f439666ef670309a4aad83e13a814ec7050ab4e29bfbca8c5cabe0e090a3ad0c'),
        2025: ('33ac9404d083099ba72ce4c4ca51800bcad8588fb9543b120751ab1551d0d7aa', 'ec39f3d242031186ea7fac23101f521e22cf4fad091cb9d2661e8f66a6f736dd')}
SOURCES = {f'growth_{y}06': (f'{LP}{y}06/{y}06.12months.growth.tot.aus.img.zip', y) for y in YEARS}
SOURCE_NOTE = 'Queensland Government LongPaddock AussieGRASS, 12months.growth.tot (kg DM/ha), CC BY 4.0'
INHERITED = ['NLUM_2010-11_mask.tif']
NCELL = 6956407
AGT = Affine(0.05, 0.0, 111.975, 0.0, -0.05, -9.975)  # AussieGRASS national grid, 840 x 680
AGH, AGW = 680, 840
TIE_TOL = 1e-9                                        # degrees: equal distances up to float rounding


def sha(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def decoded(zpath, y):
    """(array as stored, sha256 of dtype, shape, transform, CRS, nodata attribute and pixel bytes)."""
    with rasterio.open(f'zip://{zpath}!{y}06y.growth.tot.aus.img'.replace('\\', '/')) as r:
        a = r.read(1)
        assert tuple(r.transform)[:6] == tuple(AGT)[:6] and a.shape == (AGH, AGW), (y, r.transform, a.shape)
        hdr = f'{a.dtype.str}|{a.shape}|{tuple(r.transform)[:6]}|{r.crs.to_epsg() if r.crs else None}|nodata={r.nodata}|'
    return a, sha(hdr.encode() + np.ascontiguousarray(a).tobytes())


# ----------------------------------------------------------------------------- cache (shared, content-addressed)
def cache_path(cache, digest, url):
    """<cache>/<first two hex>/<raw sha256><extension of the URL's file name> (GDAL needs the .zip to read an archive)."""
    return os.path.join(cache, digest[:2], digest + os.path.splitext(url.rsplit('/', 1)[-1])[1])


def cache_index(cache):
    p = os.path.join(cache, 'index.csv')
    if not os.path.exists(p):
        return {}
    with open(p, encoding='utf-8', newline='') as f:
        return {(r['sha256'], r['url']): r for r in csv.DictReader(f)}


def cache_index_add(cache, name, url, digest, nbytes):
    rows = cache_index(cache)
    rows[(digest, url)] = dict(sha256=digest, url=url, bytes=str(nbytes), name=name)
    with open(os.path.join(cache, 'index.csv'), 'w', encoding='utf-8', newline='\n') as f:
        w = csv.DictWriter(f, fieldnames=['sha256', 'url', 'bytes', 'name'], lineterminator='\n')
        w.writeheader()
        w.writerows(rows[k] for k in sorted(rows))


def fetch(name, cache, offline, overrides, log):
    """Path of a cached copy of grid `name` whose decoded content matches its pin. Cache order: the zip retrieved at pinning,
    then any other cached copy of the same URL; else an override copy or a download. A copy whose decoded content differs
    stops the run (downloads are set aside in _quarantine); a copy whose raw bytes differ but content matches is used and
    logged. Pins are never changed here."""
    url, y = SOURCES[name]
    cpin, rpin = PINS[y]
    cands = [cache_path(cache, rpin, url)]
    cands += [cache_path(cache, d, url) for (d, u) in sorted(cache_index(cache)) if u == url and d != rpin]
    for p in cands:
        if os.path.exists(p):
            if decoded(p, y)[1] != cpin:
                raise SystemExit(f'{name}: cached file {p} does not match its content pin {cpin}; remove it and rerun')
            log.append((name, url, sha_file(p), rpin, 'cache'))
            return p
    if name in overrides:
        with open(overrides[name], 'rb') as f:
            data = f.read()
        how = 'override ' + overrides[name]
    else:
        if offline:
            raise SystemExit(f'--offline: {name} ({url}) is not in the cache {cache}')
        last = None
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (luto-pasture-productivity/1.0)'})
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                break
            except Exception as e:      # network retry
                last = e
                time.sleep(5 * (attempt + 1))
        else:
            raise SystemExit(f'{name}: download failed from {url}: {last}; supply a copy with --source {name}=PATH')
        how = 'download'
    raw = sha(data)
    p = cache_path(cache, raw, url)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    part = p[:-4] + '.part.zip'
    with open(part, 'wb') as f:
        f.write(data)
    got = decoded(part, y)[1]
    if got != cpin:
        q = os.path.join(cache, '_quarantine')
        os.makedirs(q, exist_ok=True)
        qp = os.path.join(q, f'{name}_{raw}.zip')
        os.replace(part, qp)
        raise SystemExit(f'{name}: {how} {url} has decoded content {got}, pinned {cpin} (raw sha256 {raw}, {len(data)} bytes). '
                         f'Copy set aside at {qp}; the pin is not changed.')
    os.replace(part, p)
    cache_index_add(cache, name, url, raw, len(data))
    log.append((name, url, raw, rpin, how))
    return p


# ----------------------------------------------------------------------------- layer
def layer(zips, mask_path):
    from scipy.spatial import cKDTree
    with rasterio.open(mask_path) as m:
        M = m.read(1) == 1
        T = m.transform
    rr, cc = np.nonzero(M)
    lon = T.c + (cc + 0.5) * T.a
    lat = T.f + (rr + 0.5) * T.e
    N = len(lon)
    assert N == NCELL, N
    S = np.stack([decoded(zips[f'growth_{y}06'], y)[0].astype(np.float64) for y in YEARS])
    assert S.min() >= -2 and set(np.unique(S[S < 0])) <= {-2.0}, np.unique(S[S < 0])
    assert np.isfinite(S).all()
    lake = (S == -2).any(0)
    zero_all = (S == 0).all(0)
    G = S.mean(0)
    valid = ~lake & ~zero_all                                   # 0 in every year is invalid, on land or not
    fr = (-9.975 - lat) / 0.05 - 0.5
    fc = (lon - 111.975) / 0.05 - 0.5
    i0 = np.floor(fr).astype(int)
    j0 = np.floor(fc).astype(int)
    wy = fr - i0
    wx = fc - j0
    wy = np.where(wy < 1e-9, 0., np.where(wy > 1 - 1e-9, 1., wy))
    wx = np.where(wx < 1e-9, 0., np.where(wx > 1 - 1e-9, 1., wx))
    corners = [(0, 0, (1 - wy) * (1 - wx)), (0, 1, (1 - wy) * wx), (1, 0, wy * (1 - wx)), (1, 1, wy * wx)]
    num = np.zeros(N)
    wsum = np.zeros(N)
    for di, dj, w in corners:
        ii = np.clip(i0 + di, 0, AGH - 1)
        jj = np.clip(j0 + dj, 0, AGW - 1)
        ok = valid[ii, jj] & (w > 0)
        num += np.where(ok, G[ii, jj] * w, 0.)
        wsum += np.where(ok, w, 0.)
    has = wsum > 0
    val = np.full(N, np.nan)
    val[has] = num[has] / wsum[has]
    # nearest valid pixel centre; exact-distance ties -> lowest pixel index (vi, vj are row-major from np.nonzero)
    fill = np.flatnonzero(~has)
    vi, vj = np.nonzero(valid)
    tree = cKDTree(np.c_[111.975 + 0.05 * (vj + .5), -9.975 - 0.05 * (vi + .5)])
    d, k = tree.query(np.c_[lon[fill], lat[fill]], k=16)
    tied = d <= d[:, :1] + TIE_TOL
    assert not tied[:, -1].any(), 'more than 15 equidistant valid pixels'
    choice = np.where(tied, k, np.iinfo(np.int64).max).min(1)
    val[fill] = G[vi[choice], vj[choice]]
    assert np.isfinite(val).all()
    ties = []
    for r_ in np.flatnonzero(tied.sum(1) > 1):
        px = sorted(int(x) for x in k[r_][tied[r_]])
        ties.append((int(fill[r_]), [(int(vi[x]), int(vj[x]), int(round(G[vi[x], vj[x]]))) for x in px],
                     (int(vi[choice[r_]]), int(vj[choice[r_]]))))
    info = dict(cells=N, filled_nearest=int(len(fill)), filled_with_tie=len(ties), invalid_pixels=int((~valid).sum()))
    return np.round(val).astype(np.int16), info, ties


# ----------------------------------------------------------------------------- provenance
def content_sha(a):
    a = np.ascontiguousarray(a)
    return sha(f'{a.dtype.str}|{a.shape}|'.encode() + a.tobytes())


def pkg_versions():
    out = {}
    for p in ('numpy', 'pandas', 'tables', 'rasterio', 'scipy'):
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
    ap.add_argument('--input', required=True, help='folder of the inherited 2010-build NLUM_2010-11_mask.tif (input/Nick_build/inherited_2010)')
    ap.add_argument('--out', required=True, help='output folder (e.g. input/Nick_build/pasture_productivity)')
    ap.add_argument('--cache', required=True, help='shared raw-download cache (content-addressed by sha256)')
    ap.add_argument('--offline', action='store_true', help='fail instead of downloading on a cache miss')
    ap.add_argument('--source', action='append', default=[], metavar='NAME=PATH',
                    help='a local copy of a pinned source (must match its content pin); NAME is growth_YYYY06')
    a = ap.parse_args()
    t0 = time.time()
    overrides = dict(s.split('=', 1) for s in a.source)
    assert set(overrides) <= set(SOURCES), set(overrides) - set(SOURCES)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    log = []
    zips = {k: fetch(k, a.cache, a.offline, overrides, log) for k in SOURCES}
    for name, url, raw, rpin, how in log:
        if raw != rpin:
            print(f'{name}: raw bytes {raw} differ from the copy pinned on 2026-09-28 ({rpin}); decoded content matches ({how})')
    inp = {f: os.path.join(a.input, f) for f in INHERITED}
    inp_sha = {f: sha_file(p) for f, p in inp.items()}

    pk, info, ties = layer(zips, inp['NLUM_2010-11_mask.tif'])
    s = pd.Series(pk, index=pd.Index(np.arange(NCELL, dtype=np.int64)), name='PASTURE_KG_DM_HA')
    out_h5 = os.path.join(a.out, 'pasture_kg_dm_ha.h5')
    s.to_hdf(out_h5, key='pasture_kg_dm_ha', mode='w', format='table', index=False, complevel=9)
    back = pd.read_hdf(out_h5)
    assert back.equals(s) and back.dtype == np.int16 and back.name == 'PASTURE_KG_DM_HA'
    out_nc = os.path.join(a.out, 'pasture_kg_dm_ha.nc')
    NC.write_layers(out_nc, {'PASTURE_KG_DM_HA': s.to_numpy()})
    NC.check(out_nc, {'PASTURE_KG_DM_HA': s.to_numpy()})
    print('pasture_kg_dm_ha', info, f'{time.time() - t0:.0f}s', flush=True)

    # ---- provenance (last)
    me = open(os.path.abspath(__file__), 'rb').read()
    P = []
    for name, url, raw, rpin, how in log:
        y = SOURCES[name][1]
        P.append(['source', name, url, PINS[y][0], 'sha256 of the decoded grid (dtype, shape, transform, CRS, nodata, pixels)',
                  SOURCE_NOTE])
        P.append(['source_raw', name, url, raw, 'sha256 of the raw zip used (recorded, not enforced); pinned copy '
                  + ('same' if raw == rpin else rpin), how])
    P += [['inherited', f, '', inp_sha[f], 'inherited 2010-build input (inherited_2010)', ''] for f in INHERITED]
    P += [['script', 'input/Nick_build/scripts/pasture_productivity.py', '', sha(me), 'sha256 of script', '']]
    P += [['package', k, '', v, 'version', ''] for k, v in sorted(pkg_versions().items())]
    P += [['python', sys.version.split()[0], '', '', 'interpreter', '']]
    P += [['output', 'pasture_kg_dm_ha.h5', '', sha_file(out_h5), 'sha256 of file (HDF5 bytes vary between runs)', ''],
          ['dataset', 'pasture_kg_dm_ha.h5:/pasture_kg_dm_ha values', '', content_sha(back.to_numpy()),
           'sha256 of dtype, shape and array bytes', f'{back.dtype} {back.shape}'],
          ['dataset', 'pasture_kg_dm_ha.h5:/pasture_kg_dm_ha index', '', content_sha(back.index.to_numpy()),
           'sha256 of dtype, shape and array bytes (cell order)', f'{back.index.dtype} {back.index.shape}'],
          ['netcdf', 'pasture_kg_dm_ha.nc', '', NC.content_sha(out_nc), 'sha256 of NetCDF content (nc_layout.content_sha)',
           'checked equal to the array written']]
    # nearest-filled cells with equidistant valid pixels: the tied pixels (row:col=rounded value) and the one chosen
    P += [['tie', str(c), '', '', 'tied pixels ' + ';'.join(f'{r}:{q}={v}' for r, q, v in px), f'chosen {ch[0]}:{ch[1]}']
          for c, px, ch in ties]
    write_csv(os.path.join(a.out, 'provenance.csv'), ['kind', 'item', 'url', 'sha256', 'note', 'source'], P)
    print('done', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
