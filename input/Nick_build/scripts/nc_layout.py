"""NetCDF copies of the 2021-base inputs in the layout of the jinzhu branch's input files.

Every file: NETCDF4, one dimension `cell` over the 6,956,407 land cells in the order of the .h5 rows (no cell coordinate),
zlib level 9, one chunk per column over all cells; floating-point variables keep a NaN fill value, integer variables have
none. The forms:
  - single layers and frames of 1-D columns (write_layers): one variable per column over `cell`, named as the column;
  - wide tables (write_wide): one 2-D variable named after the file stem in capitals, dims (column, cell), and one text
    coordinate per column level on `column` in the .h5 level order (integer labels, such as `year`, stay integers);
  - land-use eligibility (write_xmrj): `X_MRJ` (lm, cell, lu), stored as int8 marked dtype=bool, with text coordinates
    `lm` and `lu`.
The .h5, .npy and .csv outputs are written as before; these files sit alongside them.

Reproducibility is checked on content, not file bytes (NetCDF4/HDF5 files carry library metadata): content_sha hashes every
variable's name, dimensions, decoded dtype and shape, the attributes the writer sets, and the decoded values (text labels
joined), column by column; check() reads a file back and compares every variable with the arrays it was written from.
No luto imports.
"""
import hashlib
import os

import numpy as np
import pandas as pd
import xarray as xr

NCELL = 6956407


def _enc(dtype, chunks):
    e = dict(zlib=True, complevel=9, chunksizes=chunks)
    e['_FillValue'] = np.nan if np.issubdtype(dtype, np.floating) else None
    return e


def write_layers(path, cols):
    """cols: {variable name: 1-D array over cells, in cell order}."""
    ds = xr.Dataset({k: (('cell',), np.asarray(v)) for k, v in cols.items()})
    assert all(ds.sizes[d] == NCELL for d in ('cell',)), ds.sizes
    ds.to_netcdf(path, format='NETCDF4', encoding={k: _enc(ds[k].dtype, (NCELL,)) for k in ds.data_vars})


def write_wide(path, name, df, levels):
    """df: rows = cells (cell order), columns = the .h5 columns (MultiIndex or flat); levels: a name for each level."""
    assert len(df) == NCELL, len(df)
    cols = df.columns
    coords = {}
    for i, lv in enumerate(levels):
        lab = np.asarray(cols.get_level_values(i) if isinstance(cols, pd.MultiIndex) else cols)
        coords[lv] = (('column',), lab.astype(str) if lab.dtype.kind in 'OUS' else lab)
    da = xr.DataArray(df.to_numpy().T, dims=('column', 'cell'), coords=coords, name=name)
    da.to_dataset().to_netcdf(path, format='NETCDF4', encoding={name: _enc(da.dtype, (1, NCELL))})


def write_xmrj(path, x, lm, lu):
    """x: bool (lm, cell, lu); lm, lu: the labels."""
    assert x.dtype == bool and x.shape == (len(lm), NCELL, len(lu)), (x.dtype, x.shape)
    da = xr.DataArray(x, dims=('lm', 'cell', 'lu'), coords={'lm': np.asarray(lm, str), 'lu': np.asarray(lu, str)}, name='X_MRJ')
    da.to_dataset().to_netcdf(path, format='NETCDF4', encoding={'X_MRJ': dict(zlib=True, complevel=9, chunksizes=(1, NCELL, 1))})


def _payload(a):
    a = np.asarray(a)
    if a.dtype.kind in 'OUS':
        return '\x1f'.join(map(str, a.ravel())).encode('utf-8')
    return np.ascontiguousarray(a).tobytes()


def content_sha(path):
    """sha256 over every variable (data and coordinates, sorted by name): name, dims, decoded dtype and shape, attributes
    the writer sets (encoding attributes excluded), and the decoded values, read along the first dimension."""
    h = hashlib.sha256()
    with xr.open_dataset(path) as ds:
        for k in sorted(ds.variables):
            v = ds[k]
            attrs = {a: str(b) for a, b in sorted(v.attrs.items())}
            h.update(f'{k}|{v.dims}|{v.dtype.str}|{v.shape}|{attrs}|'.encode('utf-8'))
            if v.ndim >= 2:
                for i in range(v.shape[0]):
                    h.update(_payload(v[i].values))
            else:
                h.update(_payload(v.values))
    return h.hexdigest()


def check(path, expected):
    """Read the file back; every variable in `expected` ({name: array in the variable's dim order}) must equal it
    (NaN equal to NaN), with the decoded dtype of the array written."""
    with xr.open_dataset(path) as ds:
        assert set(expected) <= set(ds.data_vars), (set(expected) - set(ds.data_vars))
        for k, a in expected.items():
            v = ds[k]
            a = np.asarray(a)
            assert v.shape == a.shape and v.dtype == a.dtype, (k, v.shape, a.shape, v.dtype, a.dtype)
            rows = range(v.shape[0]) if v.ndim >= 2 else [None]
            for i in rows:
                got, want = (v.values, a) if i is None else (v[i].values, a[i])
                ok = np.array_equal(got, want, equal_nan=True) if got.dtype.kind == 'f' else np.array_equal(got, want)
                assert ok, (k, i)


def write_like(out_dir, stem, obj, levels=None, name=None):
    """Write <stem>.nc for an .h5 object by the layout rules (a Series: one variable named as the Series; a frame of 1-D
    columns: one variable per column; a frame with MultiIndex columns: one 2-D variable named `name`, default the stem in
    capitals, with `levels` naming the column levels), check it against the object, and return (path, content sha)."""
    p = os.path.join(out_dir, stem + '.nc')
    if isinstance(obj, pd.Series):
        cols = {str(obj.name): obj.to_numpy()}
        write_layers(p, cols)
        check(p, cols)
    elif isinstance(obj.columns, pd.MultiIndex):
        assert levels is not None and len(levels) == obj.columns.nlevels, levels
        nm = name or stem.upper()
        write_wide(p, nm, obj, levels)
        check(p, {nm: obj.to_numpy().T})
    else:
        cols = {str(c): obj[c].to_numpy() for c in obj.columns}
        write_layers(p, cols)
        check(p, cols)
    return p, content_sha(p)
