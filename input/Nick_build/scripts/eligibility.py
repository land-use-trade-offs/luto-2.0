#!/usr/bin/env python
"""Build LUTO's 2021-base land-use eligibility (x_mrj.npy) and its zone layers from public sources.

Sources (public, downloaded at run time, sha256-checked against pinned values):
  - ABARES NLUM v7.1 250 m ALUMV8 2020-21 raster and class table, and the 2020-21 agricultural probability surfaces (CC BY 4.0).
  - ABS ASGS Edition 3 SA2 2021 boundaries, GDA94 (CC BY 4.0).
  - SILO gridded monthly rainfall, 0.05 degree, 1961-2020 (Queensland Government, CC BY 4.0).
  - Australian Irrigation Areas v1A, vector (ABARES / NLWRA, CC BY 3.0 AU).
Inputs from the other 2021-base tools: xwalk_definition.csv, sa2_2021_cells.npy and class_alloc_<Y>_lmmap.npy for Y in
2010_11, 2015_16 and 2020_21 (nlum_map.py); the 2021-base lumap.h5 and lmmap.h5 (livestock_map.py).
Inherited 2010-build inputs, sha256 in provenance: NLUM_2010-11_mask.tif (the cell grid) and lumap.h5 (the 2010
footprint, a reported column of the occurrence table only). Not public sources alone.

Method (recipe dataprep.py:550-613):
  1. Occurrence table: every v7.1 2020-21 pixel (EPSG:3577) gets an SA2_2021 (polygon at the pixel centre, rasterised in
     EPSG:3577; agricultural pixels in no polygon take the nearest labelled pixel by distance transform) and a LUTO (lu, lm)
     by the definition crosswalk; one-to-many classes take the candidate with the highest probability at the pixel centre
     (first candidate where no surface has data, counted); grazing classes stay pseudo codes (100 natural, 101 modified).
     Rows are kept where the net pixel count (pixels less those that took the first candidate for lack of data) is at least
     4, that is 25 ha.
  2. Pivot per SA2 x (lm, lu); grazing pseudo codes expand to the livestock types present in the SA2 on the 2021-base lumap.
  3. Unallocated land is always dry-eligible and never irrigated; dry crops need mean April-November SILO rainfall
     1991-2020 >= 175 mm (bilinear at cell centres); irrigation needs an AIA v1A polygon at the cell centre or an irrigated
     cell of the 2021-base map in the 3 x 3 neighbourhood; irrigated crops (not irrigated pasture) also need the cell
     irrigated in at least one NLUM epoch, 2010-11, 2015-16 or 2020-21, as the class allocation defines land management
     (class_alloc_<Y>_lmmap.npy == 1; irrigation infrastructure rule, a change to default behaviour); every cell is
     eligible for its own (lm, lu) on the 2021-base map.
Outputs in --out: x_mrj.npy (bool, 2 x cells x 28, C order); xmrj_zone_layers.h5 (CELL_ID index; the two gates, the SILO
1961-2000 reference gate, both Apr-Nov means, AIA_V1A); NetCDF copies in the jinzhu layout (nc_layout.py): x_mrj.nc
(X_MRJ int8 marked bool, (lm, cell, lu), labels dry/irr and the 28 land uses) and xmrj_zone_layers.nc (one variable per
column); occurrence_2020_21.csv (the thresholded table);
provenance.csv (last).

Run as a plain script from the repository root (no luto imports):

    python input/Nick_build/scripts/eligibility.py --input input/Nick_build/inherited_2010 --nlum-map input/Nick_build/nlum_map --map-dir input/Nick_build/livestock_map
        --out input/Nick_build/eligibility --cache input/Nick_build/_cache [--offline] [--source NAME=PATH]

Dependencies: numpy, pandas, PyTables, xarray, netCDF4, rasterio, pyproj, geopandas, scipy. No luto imports
(nc_layout.py is a sibling module).
Memory: peak resident memory about 6.0 GB on the full grid (measured October 2026); run on a machine with at least 16
GB of RAM.
x_mrj.npy and the CSV reproduce byte for byte; the HDF5 file carries a group timestamp, so provenance records dataset hashes.
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
import zipfile
from importlib import metadata

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

import nc_layout as NC

# ----------------------------------------------------------------------------- sources and pins
ABARES = 'https://www.agriculture.gov.au/sites/default/files/documents/'
SILO = 'https://s3-ap-southeast-2.amazonaws.com/silo-open-data/Official/annual/monthly_rain/'
SILO_PINS = {  # year: sha256 of <year>.monthly_rain.nc (retrieved 2026-09-29)
    1961: 'a865397e22ebd1e3500c6e723f78e6f4394e039c708851087c70c3b1952443be',
    1962: '7d496ed5f55319d0574671440a1034952c9de7497bb3ff915fb42949ddcb302a',
    1963: 'f2117c6e4a7813022794426b4d9ea08a44b0b92f3817ffaacdf5c8e99beb9f0a',
    1964: 'c2a6c84c659c7733baba3b4b5fcf4702a7985b1a59baa5ec462ea573872baf26',
    1965: '15444bce92d396e29361010f5df727db7106895cfc6fccc953cb6fab2b29eafb',
    1966: 'd132ea89b16755d9fc5cb23002b0ce3638ba139e8144d0049220f97945dc7f79',
    1967: 'c4d57bd990cadbf36f6eb3e4a64ec722a6d59650de86b2574e0e50f80bd98501',
    1968: 'dc4603a66efe2784c503d3f2559ff581d59bd5b8b48b99201bd91dd902216d07',
    1969: 'd94fc8e5c32c233a27b5037c2567244572d5e382512766c2fc4dad102e657fa8',
    1970: 'a9f68cc6beb4e08ad0ed6d6feb4c009eb46c47058987691054114fc68c2bc863',
    1971: 'c34401972ede19778e45d3297504ff2afc65cdac01068e6a8c986d9cb261b25f',
    1972: 'a29502546b0d66c7c4fec53e472ab59fc9812ae8a25d00988df7bb8c7266e451',
    1973: 'a65fb4c5b262dfeb1868c8a471d8eedab6b161c341517ce05ed1532661029cc4',
    1974: '3683edacb99e9b70d00e8368e76e2fe6548629db6393068f9b8c45ea2ca0d530',
    1975: 'acd5e584d6c2305410bc968c171d85dea32af5fc419e42806a53dfcf03d976fa',
    1976: '23da986f8be42c195c39d4d56513c47331191e7e1841d22eb8ed04e37c6efbb7',
    1977: 'ff18f07a77eaab4ef18407d09c7b62efe0c088f69f7e20c4fea2af9a75cbac77',
    1978: 'f4e3cd6d469083b210a033ca34facfd0163cb9348f68199451f90b7b0d9aa4a7',
    1979: 'e33b88ac8c8ad474b2b94f627259df33e39326b3ecc71f6cb911d4ae3e98a4ce',
    1980: 'c028ea672a8153d0cee86c65c59e08c2ddcf02dce4946b63eb7976ea532c9a88',
    1981: 'd600d671168cf84ab59776c13ede885675d07f708b882399c41977becc20c0a4',
    1982: '875756665a9b50fe1062b91e032d2c97790bb3bab14cd2d58ea125353ca62940',
    1983: 'd9a4939e893c7fa9e757429c0b50e826b12d1a575362b3a9e8cf106073f2b06f',
    1984: '63e705fc1769246e47ad72d41a99eacfe1477d7ae80d8dce49d5b7c994d04570',
    1985: '6d0b5dd4b46e6cc9d21b3a514bdbf995bab8c8464f533b473ef761926554fec7',
    1986: 'a64335e2f8466bc87d573b5917cd55a0428d7f974cb48620cc975e672c2c2301',
    1987: '2ec021e27d55cc270692a5f299142a66760ac36a1e4886f031db7671264aff21',
    1988: '838733d22c440bdd114bac05d012c7049fe3d6a635d8a5adc33afd4e0330a2fb',
    1989: 'fbfa8759b54f09707a5504cc68b64054f7ab825e772aa380f9f14ec84028c318',
    1990: 'ccdc70088e0cd695046430ba48f08e0fc1bae8d651b76d383f885370b8652212',
    1991: 'f7032433f3c1b318bafa488f5c45240cee6e29bcfa1fe9c415562a78aacc41eb',
    1992: '1b685c40e08aea466fb6d43108b6d19470d388cc8f0440723d287c23fbbd65e8',
    1993: 'c8bc5f5579ed868fb8ac0f84c8ef88235251486a64c53f8fcef435e5094a122d',
    1994: 'db8a1fc4ee14b1eae8423525995c23f19daa1172f49fa5f0a81c0fe15ea596fd',
    1995: 'b26c5c28755eeed1d8a1b4d7b23d53db6221bbda5e6d534fc3d4f420d3c37a7e',
    1996: 'e0cf0232a0e1756847139c1ecb01d478b6c6842db9de3638da37bd4a3375e26a',
    1997: '8f88ffb4c43652314f3ef00492ac032d2f3001ed414cceea881c37953b0ac30a',
    1998: 'd7997c51342c32bc1b40950190dab4f0981f749df14e01c737e67de472df6b65',
    1999: '57d3509f7d32a66ab03284fd0b60415a9d91ccaaa175ed981562a1f288683fc1',
    2000: '584fab4e114bc2f5259f2d5e629cabd32c8e20e0c6dbdd07676bb2dbbcc37676',
    2001: 'ac0407b1b52feea098b5f8a0bc896557cfdbee5ec8e9c8582c85f8f5c2638708',
    2002: 'ccf2e3839f4592ebe4230c5147f116ec7b569ea88c9cf635cfe52f6623be0d8f',
    2003: 'cf9d1b6689c1a572c7da1ea0b90d06999bef358a3a463d19d9cb370d87c69438',
    2004: '224c6389da15e37c7286a2e033e2880a21b9b7e2ec57025c3301d786151b9b1a',
    2005: '7a6b8cfa665bf56fc8eb3ddac12c423727349c69fa842ce1df738aba4f69fb03',
    2006: '02e92b4bd1f812f00fd24a17a6247c57dd5542267e13b916eb5e859ad079ee28',
    2007: 'aab92bc87a2dcffef7b523d5e3aaebbbc78bc907f1b23253eed3dc718535ab36',
    2008: '4c1ea9904bfaaa881c9928293d9adc30d10047c3cd2abab0a4880cf50c66420d',
    2009: '0e6eb7edff1a65b98803db817b7fa51da7dda0a5ebbdee06f62b1bf4eb141a87',
    2010: '542ef908c931514ef0e9c9ccab8028c0bd990d663ef787de7f3e2e8deb188833',
    2011: 'd5f870b775f9eaddb26f3daefbebc42688ef3283fa50169737a7cfcb1af575b1',
    2012: '7fe830c6a0fbdb41d499529dcdef9a86ed1f2b659ddab61ccda40aa6af0f190f',
    2013: '79993cb06b3a0a4c61878645d5f5577486873350e5000338746ac902551a1469',
    2014: '80350406457d14a61f8b80fc9df36621b9a76d3b2a2172126359979871fb7160',
    2015: 'b1b87f5be116630f12ffbfc532845657f20b2bcdf50d1d1adc7e6b3a25199c33',
    2016: '352961a6ded852fa29ff3efd3735e77d30852d0e5ac19560ad685bb52d588041',
    2017: 'd93aa765125f63c2bbbe5c83a0e18e73f5baf7bae0a9a7c023618910fbf32371',
    2018: '0b9b9a75b76b158449f20cdf56391719dcbd299edfc091f50077a8352b7bef11',
    2019: '139ef5ab6976e01fbb271058a54439782e14e04a880faa6e695651cab47867cc',
    2020: 'b04a6ee92af261306681e37924386e513ca559343e90494ef6773008d04f2a64',
}
SOURCES = {
    'alum_2020_21': (ABARES + 'NLUM_v7_1_250m_ALUMV8_2020_21_alb_20260814.zip',
                     '6e937a1cf1e99bffc251b684cdd3f55429830fa0c7790b7191abfef995851fb7',
                     'ABARES NLUM v7.1 (release 2026-08-14), CC BY 4.0'),
    'prob_2020_21': (ABARES + 'NLUM_v7_1_250m_AgProbabilitySurfaces_2020_21_geo_20260814.zip',
                     'ce8549585962cfc824534edbd0a5de10bb407f90f9d18338f5f5a6817023cdd0',
                     'ABARES NLUM v7.1 (release 2026-08-14), CC BY 4.0'),
    'sa2_2021': ('https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/'
                 'jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA94.zip',
                 '87345a241c20d64068a012616adb74c0eada6cc47ce2a897d62d78026deaf0cc', 'ABS ASGS Edition 3 SA2 2021 GDA94, CC BY 4.0'),
    'aia_v1a': ('https://data.gov.au/data/dataset/3c934692-b670-4608-8f74-ee36583f74c1/resource/'
                '4b926ea7-3453-4194-b402-3e4b7dc76fd6/download/aia__r9ab__00211a02e_sgeo___.zip',
                '5ff17bb22f6a08c154c4b702964d6c2927aaae2fd9567a44c388a836dbaeed75',
                'Australian Irrigation Areas v1A (ABARES / NLWRA), vector, CC BY 3.0 AU'),
}
for _y, _p in SILO_PINS.items():
    SOURCES[f'silo_{_y}'] = (f'{SILO}{_y}.monthly_rain.nc', _p, 'SILO monthly_rain 0.05 deg (Queensland Government), CC BY 4.0')
INHERITED = ['NLUM_2010-11_mask.tif', 'lumap.h5']
NCELL = 6956407
LUS = ['Apples', 'Beef - modified land', 'Beef - natural land', 'Citrus', 'Cotton', 'Dairy - modified land',
       'Dairy - natural land', 'Grapes', 'Hay', 'Nuts', 'Other non-cereal crops', 'Pears', 'Plantation fruit', 'Rice',
       'Sheep - modified land', 'Sheep - natural land', 'Stone fruit', 'Sugar', 'Summer cereals', 'Summer legumes',
       'Summer oilseeds', 'Tropical stone fruit', 'Unallocated - modified land', 'Unallocated - natural land',
       'Vegetables', 'Winter cereals', 'Winter legumes', 'Winter oilseeds']
CLUS = [i for i, lu in enumerate(LUS) if not any(k in lu for k in ['Beef', 'Sheep', 'Dairy', 'Unallocated'])]
UNALLOC = [22, 23]
SPLIT = {331: [(25, 5), (18, 6)], 431: [(25, 5), (18, 6)], 334: [(27, 10), (20, 11)], 434: [(27, 10), (20, 11)],
         338: [(26, 8), (19, 9)], 438: [(26, 8), (19, 9)],
         341: [(0, 18), (11, 19), (16, 20), (21, 21)], 441: [(0, 18), (11, 19), (16, 20), (21, 21)]}
G_NAT, G_MOD = 100, 101
MIN_NET_PIXELS = 4                                   # 4 pixels, 25 ha, per (SA2, lm, lu)
PREC_MM = 175                                        # dataprep.py AVG_GROW_SEAS_PREC_GE_175_MM_YR
IRR_EPOCHS = ('2010_11', '2015_16', '2020_21')       # NLUM epochs of the irrigation infrastructure rule
STEP = 512


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
    return os.path.join(cache, digest[:2], digest + os.path.splitext(url.rsplit('/', 1)[-1])[1])


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
        last = None
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (luto-eligibility/1.0)'})
                with urllib.request.urlopen(req, timeout=600) as r:
                    data = r.read()
                break
            except Exception as e:      # network retry
                last = e
                time.sleep(5 * (attempt + 1))
        else:
            raise SystemExit(f'{name}: download failed from {url}: {last}; supply a copy with --source {name}=PATH')
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


def vsizip(zpath, member):
    return '/vsizip/' + zpath.replace('\\', '/') + '/' + member


def mask2d(mask_path):
    with rasterio.open(mask_path) as r:
        return r.read(1) == 1, r.transform


# ----------------------------------------------------------------------------- step 1: occurrence table
def lookup(xwalk_csv, alum_zip):
    """Raster value -> (lu, lm): the definition crosswalk for single-code classes, -1 for 1.2.5 / 1.3.3, grazing
    pseudo codes for 2.1.0, 3.2.0, 4.2.0; one-to-many classes are resolved per pixel. (Every 2020-21 value is covered.)"""
    LUT_lu = np.full(1000, -9, np.int16)
    LUT_lm = np.zeros(1000, np.int8)
    D = pd.read_csv(xwalk_csv, dtype=str)
    with zipfile.ZipFile(alum_zip) as z:
        ct = pd.read_csv(io.BytesIO(z.read('NLUM_v7_1_250m_ALUMV8_2020_21_alb.csv')))
    VAL = {t.split(' ')[0]: int(v) for v, t in zip(ct.Value, ct.TERTV8) if v}
    for _, r in D.iterrows():
        v = VAL.get(r.TERTV8, int(r.TERTV8.replace('.', '')))      # classes absent from the 2020-21 table never occur
        cs = str(r.luto_lu_codes).split(';')
        if len(cs) == 1:
            LUT_lu[v] = int(cs[0])
            LUT_lm[v] = 0 if (r.lm == '-' or int(cs[0]) == -1) else int(r.lm)
    for v in (125, 133):
        LUT_lu[v], LUT_lm[v] = -1, 0
    for v, cands in SPLIT.items():                 # placeholder only: every split pixel is resolved by probability below
        LUT_lu[v], LUT_lm[v] = cands[0][0], 1 if v >= 400 else 0
    LUT_lu[210], LUT_lm[210] = G_NAT, 0
    LUT_lu[320], LUT_lm[320] = G_MOD, 0
    LUT_lu[420], LUT_lm[420] = G_MOD, 1
    vals = [int(v) for v in ct.Value if v]
    assert all(LUT_lu[v] != -9 for v in vals), [v for v in vals if LUT_lu[v] == -9]
    return LUT_lu, LUT_lm


def occurrence(zips, xwalk_csv, mask_path, lu2010):
    import geopandas as gpd
    import pyproj
    import rasterio.features as rf
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
    LUT_lu, LUT_lm = lookup(xwalk_csv, zips['alum_2020_21'])
    A = rasterio.open(vsizip(zips['alum_2020_21'], 'NLUM_v7_1_250m_ALUMV8_2020_21_alb.tif'))
    AT = A.transform
    H, W = A.shape
    g = gpd.read_file('zip://' + zips['sa2_2021'].replace('\\', '/') + '!SA2_2021_AUST_GDA94.shp')
    g = g[g.geometry.notna() & ~g.geometry.is_empty].to_crs(3577).reset_index(drop=True)
    codes = g['SA2_CODE21'].astype(np.int64).to_numpy()
    SA = rf.rasterize(((geom, i + 1) for i, geom in enumerate(g.geometry)), out_shape=(H, W), transform=AT, fill=0,
                      dtype='uint16', all_touched=False)
    # agricultural pixels in no SA2 polygon take the SA2 of the nearest labelled pixel (distance transform in pixel space)
    AGCLS = (LUT_lu >= 0)
    AGCLS[list(SPLIT)] = True
    lab = SA > 0
    need_r, need_c = [], []
    for r0 in range(0, H, STEP):
        n = min(STEP, H - r0)
        a = A.read(1, window=Window(0, r0, W, n))
        rr, cc = np.nonzero((a != 0) & ~lab[r0:r0 + n] & AGCLS[np.clip(a, 0, 999)])
        need_r.append(rr + r0)
        need_c.append(cc)
    nr = np.concatenate(need_r)
    nc = np.concatenate(need_c)
    bnd = lab & ~ndi.binary_erosion(lab, structure=np.ones((3, 3), bool), border_value=1)
    br, bc = np.nonzero(bnd)
    del bnd, lab
    dd, _ = cKDTree(np.c_[br, bc]).query(np.c_[nr, nc], k=1)
    fillv = np.zeros(nr.size, np.int64)
    for q in range(nr.size):                       # exact distance-transform pick on a window holding the nearest labels
        R = int(np.ceil(dd[q])) + 2
        ra, rb = max(nr[q] - R, 0), min(nr[q] + R + 1, H)
        ca, cb = max(nc[q] - R, 0), min(nc[q] + R + 1, W)
        win = SA[ra:rb, ca:cb]
        idx = ndi.distance_transform_edt(win == 0, return_distances=False, return_indices=True)
        fillv[q] = win[idx[0][nr[q] - ra, nc[q] - ca], idx[1][nr[q] - ra, nc[q] - ca]]
    assert (fillv > 0).all()
    SA[nr, nc] = fillv.astype(SA.dtype)
    # probability surfaces for the one-to-many classes
    with zipfile.ZipFile(zips['prob_2020_21']) as z:
        names = z.namelist()
    P = {}
    for u in sorted({u for c in SPLIT.values() for _, u in c}):
        f = [x for x in names if x.startswith('NLUM_v7_1_probSurf_2020_21_') and x.endswith('.tif') and x.split('_')[7] == str(u)]
        assert len(f) == 1, (u, f)
        P[u] = rasterio.open(vsizip(zips['prob_2020_21'], f[0]))
    PT = next(iter(P.values())).transform
    PH, PW = next(iter(P.values())).shape
    m, LT = mask2d(mask_path)
    cellid = np.full(m.shape, -1, np.int64)
    cellid[m] = np.arange(NCELL)
    fp = lu2010 >= 0
    tr = pyproj.Transformer.from_crs(3577, 4283, always_xy=True)
    acc = {}
    nflag = 0
    colx = AT.c + (np.arange(W) + 0.5) * AT.a
    for r0 in range(0, H, STEP):
        n = min(STEP, H - r0)
        a = A.read(1, window=Window(0, r0, W, n))
        rr, cc = np.nonzero(a != 0)
        if rr.size == 0:
            continue
        v = a[rr, cc].astype(np.int64)
        s = SA[r0 + rr, cc].astype(np.int64)
        lu = LUT_lu[v].copy()
        lm = LUT_lm[v].copy()
        noprob = np.zeros(v.size, bool)
        assert (lu != -9).all(), np.unique(v[lu == -9])
        x = colx[cc]
        y = AT.f + (r0 + rr + 0.5) * AT.e
        lon, lat = tr.transform(x, y)
        for cls, cands in SPLIT.items():
            k = np.flatnonzero(v == cls)
            if k.size == 0:
                continue
            pr = np.floor((lat[k] - PT.f) / PT.e).astype(int)
            pc = np.floor((lon[k] - PT.c) / PT.a).astype(int)
            ok = (pr >= 0) & (pr < PH) & (pc >= 0) & (pc < PW)
            r_lo, r_hi = pr[ok].min(), pr[ok].max() + 1
            c_lo, c_hi = pc[ok].min(), pc[ok].max() + 1
            vals = []
            for _, u in cands:
                blk = P[u].read(1, window=Window(c_lo, r_lo, c_hi - c_lo, r_hi - r_lo)).astype(np.float32)
                q = np.full(k.size, np.nan, np.float32)
                q[ok] = blk[pr[ok] - r_lo, pc[ok] - c_lo]
                q[q == 32767] = np.nan
                vals.append(q)
            V = np.stack(vals)
            valid = ~np.isnan(V).all(0)
            best = np.argmax(np.where(np.isnan(V), -1, V), 0)
            lu[k] = np.array([c for c, _ in cands])[best]
            lm[k] = 1 if cls >= 400 else 0
            nflag += int((~valid).sum())
            noprob[k[~valid]] = True
        lr = np.floor((lat - LT.f) / LT.e).astype(np.int64)
        lc = np.floor((lon - LT.c) / LT.a).astype(np.int64)
        inb = (lr >= 0) & (lr < m.shape[0]) & (lc >= 0) & (lc < m.shape[1])
        cid = np.full(v.size, -1, np.int64)
        cid[inb] = cellid[lr[inb], lc[inb]]
        infp = np.zeros(v.size, bool)
        infp[cid >= 0] = fp[cid[cid >= 0]]
        assert not ((s == 0) & (lu >= 0)).any()
        keep = (s > 0) & (lu >= 0)
        key = (s[keep] * 2 + lm[keep]) * 1000 + lu[keep]
        for kk, ww in ((key, 0), (key[infp[keep]], 1), (key[noprob[keep]], 2)):
            uq, ct = np.unique(kk, return_counts=True)
            for q_, c_ in zip(uq.tolist(), ct.tolist()):
                d = acc.setdefault(q_, [0, 0, 0])
                d[ww] += c_
    rows = [(int(codes[(k // 1000) // 2 - 1]), (k // 1000) % 2, k % 1000, c[0], c[1], c[2]) for k, c in acc.items()]
    d = pd.DataFrame(rows, columns=['SA2_CODE21', 'lm', 'lu', 'pixels', 'pixels_in_2010_footprint',
                                    'pixels_noprob_first_candidate']).sort_values(['SA2_CODE21', 'lm', 'lu'])
    d = d[(d.pixels - d.pixels_noprob_first_candidate) > 0]                  # rows carried only by first-candidate pixels
    d = d[(d.pixels - d.pixels_noprob_first_candidate) >= MIN_NET_PIXELS]
    return d, dict(filled_ag_pixels=int(nr.size), split_pixels_no_valid_prob=nflag, rows=len(d))


# ----------------------------------------------------------------------------- step 3: zone layers
def silo_layers(zips, mask_path):
    """Mean over calendar years of the April-November SILO total (ocean NaN -> 0), bilinear at LUTO cell centres."""
    import xarray as xr
    m, _ = mask2d(mask_path)
    rr, cc = np.where(m)

    def season_mean(a, b):
        acc = None
        for y in range(a, b + 1):
            d = xr.open_dataset(zips[f'silo_{y}'])
            x = np.nan_to_num(d.monthly_rain.values[3:11].sum(0))
            d.close()
            acc = x if acc is None else acc + x
        return acc / (b - a + 1)
    fi = (-10.02 - 0.01 * rr + 44.0) / 0.05
    fj = (112.93 + 0.01 * cc - 112.0) / 0.05
    i0 = np.floor(fi).astype(int)
    j0 = np.floor(fj).astype(int)
    wi = fi - i0
    wj = fj - j0
    out = {}
    for a, b in [(1961, 2000), (1991, 2020)]:
        S = season_mean(a, b)
        v = (S[i0, j0] * (1 - wi) * (1 - wj) + S[i0 + 1, j0] * wi * (1 - wj) + S[i0, j0 + 1] * (1 - wi) * wj
             + S[i0 + 1, j0 + 1] * wi * wj).astype(np.float32)
        assert np.isfinite(v).all()
        out[(a, b)] = v
    return out


def aia_cells(aia_zip, mask_path):
    """AIA v1A, all polygons, cell-centre rule (rasterize all_touched=False) on the LUTO grid."""
    import geopandas as gpd
    import rasterio.features as rf
    m, T = mask2d(mask_path)
    g = gpd.read_file('zip://' + aia_zip.replace('\\', '/') + '!irrv1ac.shp')
    R = rf.rasterize(((geom, 1) for geom in g.geometry), out_shape=m.shape, transform=T, all_touched=False, dtype='uint8')
    return (R[m] > 0).astype(np.int8)


# ----------------------------------------------------------------------------- step 2 and 3: x_mrj
def build_xmrj(occ, sa2, lu, lm, prec, aia, mask_path, irr_ever):
    from scipy import ndimage as ndi
    codes, inv = np.unique(sa2, return_inverse=True)
    S = len(codes)
    m2, _ = mask2d(mask_path)
    I2 = np.zeros(m2.shape, bool)
    I2[m2] = (lm == 1) & (lu >= 0)
    pia = ((aia > 0) | ndi.binary_dilation(I2, ndi.generate_binary_structure(2, 2))[m2]).astype(np.int8)
    TYPES = {'beef': (1, 2), 'dairy': (5, 6), 'sheep': (14, 15)}          # (modified, natural)
    present = {t: np.bincount(inv[np.isin(lu, v)], minlength=S) > 0 for t, v in TYPES.items()}
    pos = {int(c): i for i, c in enumerate(codes)}
    occ = occ.copy()
    occ['s'] = occ.SA2_CODE21.map(pos)
    occ = occ[occ.s.notna()].astype({'s': int})
    Pn = np.zeros((2, S, 28), bool)
    base = occ[occ.lu < 28]
    Pn[base.lm.values, base.s.values, base.lu.values] = True
    graz = occ[occ.lu >= 100]
    for (lmv, g), grp in graz.groupby(['lm', 'lu']):
        for t, (mod, nat) in TYPES.items():
            ss = grp.s.values[present[t][grp.s.values]]
            Pn[lmv, ss, nat if g == G_NAT else mod] = True
    X = Pn[:, inv, :].copy()
    X[0][:, UNALLOC] = True
    X[1][:, UNALLOC] = False
    X[0][:, CLUS] &= prec.astype(bool)[:, None]
    X[1] &= pia.astype(bool)[:, None]
    irr_before = int(X[1][:, CLUS].sum())
    X[1][:, CLUS] &= irr_ever[:, None]                                     # irrigated crops: cells irrigated in an NLUM epoch
    irr_after = int(X[1][:, CLUS].sum())
    a = np.where(lu >= 0)[0]
    forced = int((~X[lm[a], a, lu[a]]).sum())
    X[lm[a], a, lu[a]] = True                                              # every cell eligible for its own (lm, lu)
    assert X[lm[a], a, lu[a]].all() and X.dtype == np.bool_ and X.shape == (2, NCELL, 28)
    return X, pia, dict(forced_before_reconcile=forced, pia_cells=int(pia.sum()), true_entries=int(X.sum()),
                        irrigated_epoch_cells=int(irr_ever.sum()), irrigated_crop_entries_before_epoch_rule=irr_before,
                        irrigated_crop_entries_after_epoch_rule=irr_after)


# ----------------------------------------------------------------------------- provenance
def content_sha(a):
    a = np.ascontiguousarray(a)
    return sha(f'{a.dtype.str}|{a.shape}|'.encode() + a.tobytes())


def pkg_versions():
    out = {}
    for p in ('numpy', 'pandas', 'tables', 'xarray', 'netCDF4', 'rasterio', 'pyproj', 'geopandas', 'pyogrio', 'scipy'):
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
    ap.add_argument('--map-dir', required=True, help='folder holding the 2021-base lumap.h5 and lmmap.h5 (livestock_map.py)')
    ap.add_argument('--out', required=True, help='output folder (e.g. input/Nick_build/eligibility)')
    ap.add_argument('--cache', required=True, help='shared raw-download cache (content-addressed by sha256)')
    ap.add_argument('--offline', action='store_true', help='fail instead of downloading on a cache miss')
    ap.add_argument('--source', action='append', default=[], metavar='NAME=PATH',
                    help='a local copy of a pinned source (must match its pin); NAME is one of the SOURCES keys')
    a = ap.parse_args()
    t0 = time.time()
    overrides = dict(s.split('=', 1) for s in a.source)
    assert set(overrides) <= set(SOURCES), set(overrides) - set(SOURCES)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    zips = {k: fetch(k, a.cache, a.offline, overrides) for k in SOURCES}
    inp = {f: os.path.join(a.input, f) for f in INHERITED}
    upstream = {'xwalk_definition.csv': os.path.join(a.nlum_map, 'xwalk_definition.csv'),
                'sa2_2021_cells.npy': os.path.join(a.nlum_map, 'sa2_2021_cells.npy'),
                'lumap.h5 (2021 base)': os.path.join(a.map_dir, 'lumap.h5'), 'lmmap.h5 (2021 base)': os.path.join(a.map_dir, 'lmmap.h5')}
    upstream.update({f'class_alloc_{y}_lmmap.npy': os.path.join(a.nlum_map, f'class_alloc_{y}_lmmap.npy') for y in IRR_EPOCHS})
    in_sha = {f: sha_file(p) for f, p in {**inp, **upstream}.items()}
    mask_path = inp['NLUM_2010-11_mask.tif']
    lu2010 = pd.read_hdf(inp['lumap.h5']).to_numpy().astype(np.int64)

    occ, oinfo = occurrence(zips, upstream['xwalk_definition.csv'], mask_path, lu2010)
    occ_path = os.path.join(a.out, 'occurrence_2020_21.csv')
    occ.to_csv(occ_path, index=False)
    occ = pd.read_csv(occ_path)
    print('occurrence', oinfo, f'{time.time() - t0:.0f}s', flush=True)
    silo = silo_layers(zips, mask_path)
    prec = (silo[(1991, 2020)] >= PREC_MM).astype(np.int8)
    prec61 = (silo[(1961, 2000)] >= PREC_MM).astype(np.int8)
    aia = aia_cells(zips['aia_v1a'], mask_path)
    print('zone layers', f'{time.time() - t0:.0f}s', flush=True)
    lu = pd.read_hdf(upstream['lumap.h5 (2021 base)']).to_numpy().astype(np.int64)
    lm = pd.read_hdf(upstream['lmmap.h5 (2021 base)']).to_numpy().astype(np.int64)
    sa2 = np.load(upstream['sa2_2021_cells.npy']).astype(np.int64)
    irr_ever = np.zeros(NCELL, bool)
    for y in IRR_EPOCHS:
        e = np.load(upstream[f'class_alloc_{y}_lmmap.npy'])
        assert e.shape == (NCELL,)
        irr_ever |= e == 1
    X, pia, xinfo = build_xmrj(occ, sa2, lu, lm, prec, aia, mask_path, irr_ever)
    np.save(os.path.join(a.out, 'x_mrj.npy'), X)
    z = pd.DataFrame({'AVG_GROW_SEAS_PREC_GE_175_MM_YR': prec, 'POTENTIAL_IRRIGATION_AREAS': pia,
                      'AVG_GROW_SEAS_PREC_GE_175_MM_YR_SILO_1961_2000': prec61,
                      'AVG_GROW_SEAS_PREC_APR_NOV_MM_SILO_1991_2020': silo[(1991, 2020)].astype(np.float32),
                      'AVG_GROW_SEAS_PREC_APR_NOV_MM_SILO_1961_2000': silo[(1961, 2000)].astype(np.float32),
                      'AIA_V1A': aia})
    z.index.name = 'CELL_ID'
    zpath = os.path.join(a.out, 'xmrj_zone_layers.h5')
    z.to_hdf(zpath, key='xmrj_zone_layers', mode='w', format='table', index=True, complevel=9)
    assert (z.index.to_numpy() == np.arange(NCELL)).all()
    xnc = os.path.join(a.out, 'x_mrj.nc')                                              # NetCDF copies, jinzhu layout
    NC.write_xmrj(xnc, X, ['dry', 'irr'], LUS)
    NC.check(xnc, {'X_MRJ': X})
    ncs = [(xnc, NC.content_sha(xnc)), NC.write_like(a.out, 'xmrj_zone_layers', z)]
    print('x_mrj', xinfo, f'{time.time() - t0:.0f}s', flush=True)

    # ---- provenance (last)
    me = open(os.path.abspath(__file__), 'rb').read()
    P = [['source', k, url, pin, 'sha256 of raw file', note] for k, (url, pin, note) in SOURCES.items()]
    P += [['inherited', f, '', in_sha[f], 'inherited 2010-build input (inherited_2010)', ''] for f in INHERITED]
    P += [['upstream', f, '', in_sha[f], 'input from another 2021-base tool', p.replace('\\', '/')] for f, p in upstream.items()]
    P += [['script', 'input/Nick_build/scripts/eligibility.py', '', sha(me), 'sha256 of script', '']]
    P += [['package', k, '', v, 'version', ''] for k, v in sorted(pkg_versions().items())]
    P += [['python', sys.version.split()[0], '', '', 'interpreter', '']]
    P += [['output', f, '', sha_file(os.path.join(a.out, f)), 'sha256', ''] for f in ('x_mrj.npy', 'occurrence_2020_21.csv')]
    back = pd.read_hdf(zpath)
    P += [['output', 'xmrj_zone_layers.h5', '', sha_file(zpath), 'sha256 of file (HDF5 bytes vary between runs)', '']]
    P += [['dataset', f'xmrj_zone_layers.h5:/xmrj_zone_layers {c}', '', content_sha(back[c].to_numpy()),
           'sha256 of dtype, shape and array bytes', str(back[c].dtype)] for c in back.columns]
    P += [['dataset', 'xmrj_zone_layers.h5:/xmrj_zone_layers index', '', content_sha(back.index.to_numpy()),
           'sha256 of dtype, shape and array bytes (cell order)', back.index.name]]
    P += [['netcdf', os.path.basename(p), '', h, 'sha256 of NetCDF content (nc_layout.content_sha)', 'checked equal to the arrays written'] for p, h in ncs]
    P += [['info', k, '', '', str(v), ''] for k, v in {**oinfo, **xinfo}.items()]
    write_csv(os.path.join(a.out, 'provenance.csv'), ['kind', 'item', 'url', 'sha256', 'note', 'source'], P)
    print('done', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
