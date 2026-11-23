#!/usr/bin/env python
"""Fill missing climate-change-impact multipliers on the 2021-base map: the 8 climate_change_impacts_<rcp>_CO2_FERT_<ON|OFF>.h5.

Sources: none public. The multipliers cannot be regenerated from a public source, so they are filled, not
rebuilt. Inherited 2010-build inputs from --input (inherited_2010/, copies of 2010-build files), each sha256 recorded in the provenance file: the 8
climate_change_impacts_*.h5 of the same names, real_area.h5, state_id.npy, ag_landuses.csv. Inputs from the other
2021-base tools: lumap.h5 and lmmap.h5 (livestock_map.py), x_mrj.npy (eligibility.py) and the SA2
2021 cell key sa2_2021_cells.npy (nlum_map.py). The outputs are therefore not built from public sources alone.

Method: target cells of a (lm, lu) column = cells allocated to (lm, lu) on the map OR eligible for
it in x_mrj. A NaN on a target cell (per column lm, lu, year) takes the real-area-weighted mean of that column over the
non-NaN cells of the same SA2_2021; if none, the same state; if none, national. Non-target cells and non-NaN values are
unchanged (bit-identical, checked as uint32 views). No columns are added: (dry, Pears) has none, and LUTO returns ones for
a (lm, lu) without columns. The input is read in row chunks in two passes (weighted sums, then fill and append).
Outputs in --out: the 8 files in the layout of the 2010-base files (pandas table, same key, zlib 9, no data columns,
CELL_ID index, 141 float32 columns in the same order) and a NetCDF copy of each in the jinzhu layout
(CLIMATE_CHANGE_IMPACT over (column, cell), levels lm, lu, year; nc_layout.py; each file is read back whole for it);
cci_fill_by_column.csv, cci_fill_summary.csv;
provenance.csv (last).

Run as a plain script from the repository root (no luto imports):

    python input/Nick_build/scripts/cci_fill.py --input input/Nick_build/inherited_2010 --livestock-map input/Nick_build/livestock_map --x-mrj input/Nick_build/eligibility/x_mrj.npy
        --sa2 input/Nick_build/nlum_map/sa2_2021_cells.npy --out input/Nick_build/cci_fill

Dependencies: numpy, pandas, PyTables, xarray, netCDF4. No luto imports (nc_layout.py is a sibling module).
Memory: peak resident memory 8.7 GiB on the full grid, run time 69 minutes, both with the NetCDF copies (measured 22 October
2026 in a full input build: resident set size of the process and its children, sampled every 0.5 s with psutil on
Windows); run on a machine with at least 16 GB of RAM.
Output values are deterministic (fixed chunk order, no randomness); HDF5 file bytes carry a group timestamp, so provenance
records dataset content hashes.
"""
import argparse
import csv
import hashlib
import os
import sys
import time
from importlib import metadata

import numpy as np
import pandas as pd

import nc_layout as NC

SUFFIX = [f'rcp{r}_CO2_FERT_{c}' for r in ('2p6', '4p5', '6p0', '8p5') for c in ('ON', 'OFF')]
INHERITED = [f'climate_change_impacts_{s}.h5' for s in SUFFIX] + ['real_area.h5', 'state_id.npy', 'ag_landuses.csv']
NCELL = 6956407
NCOL = 141
LMN = {'dry': 0, 'irr': 1}
CHUNK = 250000
NSTATE = 10
SUMMARY_FILE = 'rcp4p5_CO2_FERT_OFF'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


class ContentSha:
    """sha256 of dtype|shape|array bytes, fed in row chunks (equal to hashing the whole C-ordered array)."""
    def __init__(self, dtype, shape):
        self.h = hashlib.sha256(f'{np.dtype(dtype).str}|{shape}|'.encode())

    def update(self, a):
        self.h.update(np.ascontiguousarray(a).tobytes())

    def hexdigest(self):
        return self.h.hexdigest()


def fill_one(src, dst, area, state, si, NS, target, pairs_of, lus, lu, lm):
    with pd.HDFStore(src, mode='r') as s:
        key = s.keys()[0].lstrip('/')
        n_in = s.get_storer(key).nrows
    cols = pd.read_hdf(src, start=0, stop=1).columns
    nc = len(cols)
    assert n_in == NCELL and nc == NCOL, (n_in, nc)
    pairs = pairs_of(cols)
    N = NCELL
    # pass 1: weighted sums per SA2, state and national, per column
    s_sa = np.zeros((NS, nc))
    w_sa = np.zeros((NS, nc))
    s_st = np.zeros((NSTATE, nc))
    w_st = np.zeros((NSTATE, nc))
    for a in range(0, N, CHUNK):
        b = min(a + CHUNK, N)
        A = pd.read_hdf(src, start=a, stop=b).to_numpy()
        nn = ~np.isnan(A)
        w = area[a:b]
        for c in range(nc):
            wc = w * nn[:, c]
            v = np.where(nn[:, c], A[:, c], 0) * wc
            s_sa[:, c] += np.bincount(si[a:b], v, NS)
            w_sa[:, c] += np.bincount(si[a:b], wc, NS)
            s_st[:, c] += np.bincount(state[a:b], v, NSTATE)
            w_st[:, c] += np.bincount(state[a:b], wc, NSTATE)
    with np.errstate(invalid='ignore', divide='ignore'):
        m_sa = np.where(w_sa > 0, s_sa / w_sa, np.nan)
        m_st = np.where(w_st > 0, s_st / w_st, np.nan)
        m_na = s_sa.sum(0) / w_sa.sum(0)
    # pass 2: fill and append; check unchanged values and target coverage on the fly
    st = {c: dict(target=0, nan=0, sa2=0, state=0, natl=0, area=0.0, alloc=0) for c in range(nc)}
    ok = dict(unchanged=True, zero_nan_target=True, nonfilled_nan_kept=True, index=True)
    if os.path.exists(dst):
        os.remove(dst)
    with pd.HDFStore(dst, mode='w', complevel=9, complib='zlib') as o:
        for a in range(0, N, CHUNK):
            b = min(a + CHUNK, N)
            D = pd.read_hdf(src, start=a, stop=b)
            ok['index'] &= bool((D.index.values == np.arange(a, b)).all())
            A0 = D.to_numpy()
            A = A0.copy()
            for c in range(nc):
                m, l = pairs[c]
                T = target(m, l)[a:b]
                bad = T & np.isnan(A[:, c])
                q = st[c]
                q['target'] += int(T.sum())
                if bad.any():
                    i = np.where(bad)[0]
                    v1 = m_sa[si[a:b][i], c]
                    v2 = m_st[state[a:b][i], c]
                    v = np.where(~np.isnan(v1), v1, np.where(~np.isnan(v2), v2, m_na[c]))
                    lev = np.where(~np.isnan(v1), 0, np.where(~np.isnan(v2), 1, 2))
                    assert not np.isnan(v).any()
                    A[i, c] = v.astype(np.float32)
                    q['nan'] += len(i)
                    q['sa2'] += int((lev == 0).sum())
                    q['state'] += int((lev == 1).sum())
                    q['natl'] += int((lev == 2).sum())
                    q['area'] += float(area[a:b][i].sum())
                    q['alloc'] += int(((lu[a:b][i] == lus.index(l)) & (lm[a:b][i] == m)).sum())
                na = np.isnan(A0[:, c])
                ok['zero_nan_target'] &= not np.isnan(A[T, c]).any()
                ok['nonfilled_nan_kept'] &= bool(np.isnan(A[na & ~T, c]).all())
            nn = ~np.isnan(A0)
            ok['unchanged'] &= bool(np.array_equal(A0[nn].view(np.uint32), A[nn].view(np.uint32)))
            o.append(key, pd.DataFrame(A, index=D.index, columns=D.columns), format='table', index=False, data_columns=None)
            del A, A0, D
    assert all(ok.values()), ok
    rows = [dict(lm=cols[c][0], lu=cols[c][1], year=cols[c][2], target_cells=st[c]['target'], nan_target_cells=st[c]['nan'],
                 filled_sa2=st[c]['sa2'], filled_state=st[c]['state'], filled_national=st[c]['natl'],
                 area_filled_ha=st[c]['area'], allocated_cells_filled=st[c]['alloc']) for c in range(nc)]
    return key, cols, rows


def dataset_hashes(path, key, cols):
    """Content sha256 per column, of all values and of the index, read back from the written file in row chunks."""
    hc = [ContentSha(np.float32, (NCELL,)) for _ in cols]
    hv = ContentSha(np.float32, (NCELL, len(cols)))
    hi = None
    for a in range(0, NCELL, CHUNK):
        D = pd.read_hdf(path, key=key, start=a, stop=min(a + CHUNK, NCELL))
        assert (D.dtypes == np.float32).all()
        if hi is None:
            hi = ContentSha(D.index.dtype, (NCELL,))
        X = D.to_numpy()
        for c in range(len(cols)):
            hc[c].update(X[:, c])
        hv.update(X)
        hi.update(D.index.to_numpy())
    return [h.hexdigest() for h in hc], hv.hexdigest(), hi.hexdigest()


def pkg_versions():
    out = {}
    for p in ('numpy', 'pandas', 'tables', 'xarray', 'netCDF4'):
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
    ap.add_argument('--livestock-map', required=True, help='output folder of livestock_map.py (lumap.h5, lmmap.h5)')
    ap.add_argument('--x-mrj', required=True, help='x_mrj.npy from eligibility.py')
    ap.add_argument('--sa2', required=True, help='sa2_2021_cells.npy from nlum_map.py')
    ap.add_argument('--out', required=True, help='output folder (e.g. input/Nick_build/cci_fill)')
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    inp = {f: os.path.join(a.input, f) for f in INHERITED}
    upstream = {'lumap.h5': os.path.join(a.livestock_map, 'lumap.h5'), 'lmmap.h5': os.path.join(a.livestock_map, 'lmmap.h5'),
                'x_mrj.npy': a.x_mrj, 'sa2_2021_cells.npy': a.sa2}
    in_sha = {f: sha_file(p) for f, p in {**inp, **upstream}.items()}

    lus = pd.read_csv(inp['ag_landuses.csv'], header=None)[0].tolist()
    lu = pd.read_hdf(upstream['lumap.h5']).to_numpy()
    lm = pd.read_hdf(upstream['lmmap.h5']).to_numpy()
    x = np.load(upstream['x_mrj.npy'], mmap_mode='r')
    area = np.nan_to_num(np.asarray(pd.read_hdf(inp['real_area.h5']).to_numpy(), float).ravel())
    state = np.load(inp['state_id.npy'])
    sa2 = np.load(upstream['sa2_2021_cells.npy'])
    assert len(lu) == len(lm) == len(area) == len(state) == len(sa2) == x.shape[1] == NCELL
    _, si = np.unique(sa2, return_inverse=True)
    si = si.ravel()
    NS = int(si.max()) + 1
    tcache = {}

    def target(m, l):
        if (m, l) not in tcache:
            j = lus.index(l)
            tcache[(m, l)] = np.asarray(x[m, :, j]).astype(bool) | ((lu == j) & (lm == m))
        return tcache[(m, l)]

    def pairs_of(cols):
        return [(LMN[c[0]], c[1]) for c in cols]

    R, outs = [], {}
    for sfx in SUFFIX:
        f = f'climate_change_impacts_{sfx}.h5'
        key, cols, rows = fill_one(inp[f], os.path.join(a.out, f), area, state, si, NS, target, pairs_of, lus, lu, lm)
        R += [dict(file=sfx, **r) for r in rows]
        outs[f] = (key, cols)
        print(sfx, 'filled', sum(r['nan_target_cells'] for r in rows), f'{time.time() - t0:.0f}s', flush=True)
    ncs = []                                                                       # NetCDF copies, jinzhu layout
    for f, (key, _) in outs.items():
        D = pd.read_hdf(os.path.join(a.out, f), key=key)
        ncs.append(NC.write_like(a.out, f[:-3], D, ['lm', 'lu', 'year'], name='CLIMATE_CHANGE_IMPACT'))
        del D
    R = pd.DataFrame(R)
    R.to_csv(os.path.join(a.out, 'cci_fill_by_column.csv'), index=False, lineterminator='\n')
    g = R[R.file == SUMMARY_FILE].groupby(['lm', 'lu']).agg(
        target_cells=('target_cells', 'max'), nan_cells_max_year=('nan_target_cells', 'max'),
        filled_sa2_colsum=('filled_sa2', 'sum'), filled_state_colsum=('filled_state', 'sum'),
        filled_national_colsum=('filled_national', 'sum'), area_filled_ha_max_year=('area_filled_ha', 'max'),
        allocated_cells_filled_max_year=('allocated_cells_filled', 'max'))
    g.to_csv(os.path.join(a.out, 'cci_fill_summary.csv'), lineterminator='\n')

    # ---- provenance (last)
    me = open(os.path.abspath(__file__), 'rb').read()
    PV = [['inherited', f, '', in_sha[f], 'inherited 2010-build input (inherited_2010)', ''] for f in INHERITED]
    PV += [['upstream', f, '', in_sha[f], 'input from another 2021-base tool', upstream[f].replace('\\', '/')] for f in upstream]
    PV += [['script', 'input/Nick_build/scripts/cci_fill.py', '', sha(me), 'sha256 of script', '']]
    PV += [['package', k, '', v, 'version', ''] for k, v in sorted(pkg_versions().items())]
    PV += [['python', sys.version.split()[0], '', '', 'interpreter', '']]
    for f, (key, cols) in outs.items():
        p = os.path.join(a.out, f)
        with pd.HDFStore(p, mode='r') as s:
            st = s.get_storer(key)
            fl = st.table.filters
            assert st.nrows == NCELL and fl.complevel == 9 and fl.complib == 'zlib'
        hc, hv, hi = dataset_hashes(p, key, cols)
        PV.append(['output', f, '', sha_file(p), 'sha256 of file (HDF5 bytes vary between runs)', ''])
        PV += [['dataset', f'{f}:/{key} values {c}', '', h, 'sha256 of dtype, shape and array bytes', ''] for c, h in zip(cols, hc)]
        PV.append(['dataset', f'{f}:/{key} values', '', hv, 'sha256 of dtype, shape and array bytes', f'{len(cols)} columns float32 ({NCELL}, {len(cols)})'])
        PV.append(['dataset', f'{f}:/{key} index', '', hi, 'sha256 of dtype, shape and array bytes', 'cell order'])
        print(f, 'hashed', f'{time.time() - t0:.0f}s', flush=True)
    for f in ('cci_fill_by_column.csv', 'cci_fill_summary.csv'):
        PV.append(['output', f, '', sha_file(os.path.join(a.out, f)), 'sha256', ''])
    PV += [['netcdf', os.path.basename(p), '', h, 'sha256 of NetCDF content (nc_layout.content_sha)', 'checked equal to the .h5 arrays'] for p, h in ncs]
    write_csv(os.path.join(a.out, 'provenance.csv'), ['kind', 'item', 'url', 'sha256', 'note', 'source'], PV)
    print('done', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
