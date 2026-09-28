#!/usr/bin/env python
"""Build the observed ag-management adoption caps and the existing-HIR-project cell layer from public sources.

Sources (public, downloaded at run time, sha256-checked against pinned values):
  - Clean Energy Regulator (CER) ACCU Scheme project register (CSV): HIR method projects, registration dates,
    ACCU issuance by financial year, Carbon Estimation Area (CEA) mapping-file URLs.
  - CER CEA mapping files (zipped shapefiles, GDA2020), one per mapped HIR project. Their sha256 are recorded
    in cea_manifest.csv and the manifest's canonical hash is pinned.
  - ABS Land Management and Farming in Australia 2016-17 (cat. 4627.0, 46270DO001_201617), Table 1, Australia.

Outputs (into --out):
  - caps.csv: option, land_use, year_end, cap_fraction, source, status. One row per (option, land use, year);
    LUTO holds each cap flat after its last year (settings.AG_MANAGEMENT_OBSERVED_CAPS).
  - hir_project_cells.csv: cell (index into the NLUM land cells, np.nonzero order), project_id, fy_registered,
    fraction (share of the 0.01 deg cell inside the project's CEA, 5 x 5 supersampled).
  - hir_projects.csv: one row per HIR project: mapped flag, registration FY, revoked flag, CEA area (ha, rasterised),
    issuance to the baseline FY.
  - hir_scaling.csv: issuance of mapped and all projects to the baseline FY, and the scale factor applied to the
    mapped area for the unmapped projects.
  - cea_manifest.csv, provenance.csv.

Run as a plain script from the repo root (no luto imports):

    python luto/tools/ag_mgt_caps.py --out input/ag_mgt_caps --input input [--cache DIR]

--input is LUTO's input folder, for NLUM_2010-11_mask.tif, lumap.h5, real_area.h5 and ag_landuses.csv.
Dependencies: pandas, numpy, geopandas (pyogrio, pyarrow), shapely, rasterio, xlrd (all in requirements.yml).
Output is deterministic.
"""
import argparse
import hashlib
import io
import os
import re
import sys
import time
import urllib.request

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely

# ----------------------------------------------------------------------------- sources and pins
CER_REGISTER_URL = 'https://cer.gov.au/document/accu-scheme-project-register-0'
ABS_4627_URL = ('https://www.abs.gov.au/statistics/industry/agriculture/land-management-and-farming-australia/'
                '2016-17/46270do001_201617.xls')
PIN = {
    'cer_register_raw': '6a10f1b5e162f6b864d204ff63ba5a753d65843a0054d83be2ece7fce6bc3ce4',
    'abs_4627_raw': '7ee0e73a3bdc12c97d1c70dcdafbd4cfc422e9c3a299edef46d11a68363de14c',
    'cea_manifest_canonical': '97e7f9a7d66d558f0f3b58eee64562578edae4c7144b33b991dc8e4c12612133',   # sha256 of sorted "project_id,url,sha256" lines (311 HIR CEA files)
}
SOURCE_DATES = {
    'CER': '2026-09-28 (retrieval date; the register is a live file with no version stamp)',
    'ABS': '2018-06-26 (release date, 46270DO001_201617 Contents sheet)',
}

# ----------------------------------------------------------------------------- design constants
HIR_METHOD = 'Human-Induced Regeneration'   # all three compilations of the 2013 determination
BASELINE_FY = 2024          # Nick ruling 4: HIR baseline and cap held flat from FY2023/24 (last NIR year)
YEARS = list(range(2010, BASELINE_FY + 1))
SUPERSAMPLE = 5             # sub-cells per 0.01 deg cell side for the CEA coverage fraction
HIR_LU = {'HIR - Beef': 'Beef - natural land', 'HIR - Sheep': 'Sheep - natural land'}
LUS_PA = ['Hay', 'Summer cereals', 'Summer legumes', 'Summer oilseeds', 'Winter cereals', 'Winter legumes', 'Winter oilseeds',
          'Cotton', 'Other non-cereal crops', 'Rice', 'Sugar', 'Vegetables',
          'Apples', 'Citrus', 'Grapes', 'Nuts', 'Pears', 'Plantation fruit', 'Stone fruit', 'Tropical stone fruit']
LUS_BIOCHAR = [lu for lu in LUS_PA if lu not in ('Cotton', 'Other non-cereal crops', 'Rice', 'Sugar', 'Vegetables')]
ABS_YEAR = 2017             # 2016-17, FY ending 2017
ABS_CROP_AREA = 'Land use - Land mainly used for crops - Area (ha) (d)'
ABS_PA_ITEMS = ('Fertiliser - Fertiliser containing nitrification inhibitor - Area applied to (ha)',
                'Fertiliser - Nitrate slow release fertiliser - Area applied to (ha)',
                'Fertiliser - Urea slow release fertiliser - Area applied to (ha)',
                'Fertiliser - Other slow release fertiliser - Area applied to (ha)')
ABS_BIOCHAR = 'Soil management - Soil enhancer use - Biochar - Area applied to (ha)'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def fetch(url, cache, name=None):
    """Bytes of url: read from cache if present, else download (and store in cache)."""
    path = os.path.join(cache, name or os.path.basename(url)) if cache else None
    if path and os.path.exists(path):
        return open(path, 'rb').read()
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (LUTO2 ag_mgt_caps.py)'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                b = r.read()
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2)
    if path:
        os.makedirs(cache, exist_ok=True)
        open(path, 'wb').write(b)
    time.sleep(0.5)
    return b


def pinned(name, b):
    if PIN[name] is not None and sha(b) != PIN[name]:
        raise SystemExit(f'sha256 mismatch for {name}: expected {PIN[name]}, got {sha(b)}')
    return b


def fy_end(date_str):
    """Financial year ending (June) of a dd/mm/yyyy date."""
    d, m, y = (int(x) for x in date_str.split('/'))
    return y + 1 if m >= 7 else y


def num(s):
    return pd.to_numeric(s.fillna('0').str.replace(',', '', regex=False).str.strip(), errors='coerce').fillna(0)


def write_csv(path, df):
    df.to_csv(path, index=False, lineterminator='\n', float_format='%.9g')


def hir_register(raw):
    """HIR projects: id, registration FY, revoked, CEA URL, issuance (KACCU + NKACCU) up to and including BASELINE_FY."""
    reg = pd.read_csv(io.BytesIO(raw), encoding='cp1252', dtype=str)
    h = reg[reg['Method'].str.contains(HIR_METHOD, case=False, na=False)].copy()
    fy_cols = {}
    for c in reg.columns:
        m = re.search(r'Units Issued in Financial Year (\d{4})/(\d{2})', c)
        if m:
            fy_cols.setdefault(int(m.group(1)) + 1, []).append(c)
    issued = sum(num(h[c]) for fy, cs in fy_cols.items() if fy <= BASELINE_FY for c in cs)
    return pd.DataFrame({
        'project_id': h['Project ID'].str.strip(),
        'fy_registered': h['Date Project Registered'].map(fy_end),
        'revoked': h['Project revoked'].str.strip().eq('Yes'),
        'cea_url': h['Carbon Estimation Area mapping file URL'].fillna('').str.strip(),
        'issued_to_baseline_fy': issued.astype('int64'),
    }).sort_values('project_id').reset_index(drop=True)


def cea_cells(geoms, transform, shape, land_index):
    """(cell, fraction) of the NLUM land cells a CEA's geometries (EPSG:4283) cover: the share of each cell's 5 x 5
    sub-cell centres that fall inside a CEA part. A point-in-polygon query on an STRtree of the parts, not a GDAL
    rasterise: one CEA has 265,902 parts, which GDAL burns in minutes; the tree takes seconds."""
    parts = shapely.get_parts(np.asarray(geoms.values))
    parts = parts[~shapely.is_empty(parts)]
    minx, miny, maxx, maxy = geoms.total_bounds
    inv = ~transform
    c0, r0 = (int(np.floor(v)) for v in inv * (minx, maxy))
    c1, r1 = (int(np.ceil(v)) for v in inv * (maxx, miny))
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, shape[0]), min(c1, shape[1])
    if r1 <= r0 or c1 <= c0:
        return np.empty(0, np.int64), np.empty(0)
    n_r, n_c = (r1 - r0) * SUPERSAMPLE, (c1 - c0) * SUPERSAMPLE
    ii, jj = np.meshgrid(np.arange(n_r), np.arange(n_c), indexing='ij')
    x, y = transform * (c0 + (jj.ravel() + 0.5) / SUPERSAMPLE, r0 + (ii.ravel() + 0.5) / SUPERSAMPLE)
    inside = np.zeros(n_r * n_c, dtype=bool)
    inside[shapely.STRtree(parts).query(shapely.points(x, y), predicate='within')[0]] = True
    frac = inside.reshape(r1 - r0, SUPERSAMPLE, c1 - c0, SUPERSAMPLE).mean(axis=(1, 3))
    rr, cc = np.nonzero(frac)
    cell = land_index[rr + r0, cc + c0]
    keep = cell >= 0                                        # drop ocean / off-mask cells
    return cell[keep], frac[rr, cc][keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='output folder (e.g. input/ag_mgt_caps)')
    ap.add_argument('--input', required=True, help="LUTO input folder (NLUM mask, lumap.h5, real_area.h5, ag_landuses.csv)")
    ap.add_argument('--cache', default=None, help='directory to cache downloaded files in')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---- LUTO grid: NLUM land cells in np.nonzero order, their base-year land use and real area ----
    with rasterio.open(os.path.join(a.input, 'NLUM_2010-11_mask.tif')) as r:
        nlum, transform, crs = r.read(1), r.transform, r.crs
    land_index = np.full(nlum.shape, -1, np.int64)
    land_index[np.nonzero(nlum == 1)] = np.arange(int((nlum == 1).sum()))
    lumap = pd.read_hdf(os.path.join(a.input, 'lumap.h5')).to_numpy()
    real_area = pd.read_hdf(os.path.join(a.input, 'real_area.h5')).to_numpy()
    ag_lus = open(os.path.join(a.input, 'ag_landuses.csv')).read().split('\n')
    ag_lus = [lu.strip() for lu in ag_lus if lu.strip()]
    if len(lumap) != len(real_area) or len(lumap) != land_index.max() + 1:
        raise SystemExit('lumap.h5, real_area.h5 and the NLUM mask disagree on the number of land cells.')

    # ---- ABS 4627.0: Table 1 Australia, items by 'Commodity description', values in 'Estimate' ----
    abs_raw = pinned('abs_4627_raw', fetch(ABS_4627_URL, a.cache, 'abs_4627_2016-17.xls'))
    t = pd.read_excel(io.BytesIO(abs_raw), sheet_name='Aust.', header=None)
    head = t.index[t.eq('Commodity description').any(axis=1)][0]
    t = t.iloc[head + 1:].set_axis([str(c).strip() for c in t.iloc[head]], axis=1)
    val = t.dropna(subset=['Commodity description']).set_index(t['Commodity description'].dropna().str.strip())['Estimate']
    crop = float(val[ABS_CROP_AREA])
    pa = sum(float(val[k]) for k in ABS_PA_ITEMS) / crop
    bc = float(val[ABS_BIOCHAR]) / crop

    # ---- CER register and CEA files ----
    reg_raw =pinned('cer_register_raw', fetch(CER_REGISTER_URL, a.cache, 'accu_project_register.csv'))
    proj = hir_register(reg_raw)
    manifest, cells, areas = [], [], {}
    for n, p in enumerate(proj[proj['cea_url'] != ''].itertuples()):
        b = fetch(p.cea_url, a.cache)
        manifest.append([p.project_id, p.cea_url, sha(b)])
        g = gpd.read_file(io.BytesIO(b), use_arrow=True).to_crs(crs)
        # no union: a CEA can be one multipolygon of millions of vertices, and rasterising burns overlapping parts once
        c, f = cea_cells(g.geometry, transform, nlum.shape, land_index)
        areas[p.project_id] = float((f * real_area[c]).sum())
        cells.append(pd.DataFrame({'cell': c, 'project_id': p.project_id, 'fy_registered': p.fy_registered, 'fraction': f}))
        if n % 25 == 0:
            print(f'  CEA {n + 1}: {p.project_id}', flush=True)
    manifest = pd.DataFrame(manifest, columns=['project_id', 'url', 'sha256']).sort_values('project_id')
    canon = '\n'.join(','.join(r) for r in manifest.itertuples(index=False)).encode()
    pinned('cea_manifest_canonical', canon)
    cells = pd.concat(cells).sort_values(['cell', 'project_id']).reset_index(drop=True)
    proj['mapped'] = proj['project_id'].isin(manifest['project_id'])
    proj['cea_area_ha'] = proj['project_id'].map(areas)

    # ---- scaling for the unmapped projects: by their share of issuance to the baseline FY (ruling 4) ----
    iss_all = int(proj['issued_to_baseline_fy'].sum())
    iss_mapped = int(proj.loc[proj['mapped'], 'issued_to_baseline_fy'].sum())
    scale = iss_all / iss_mapped
    scaling = pd.DataFrame([{'baseline_fy': BASELINE_FY, 'projects_all': len(proj), 'projects_mapped': int(proj['mapped'].sum()),
                             'issued_all': iss_all, 'issued_mapped': iss_mapped, 'scale_factor': scale}])

    # ---- caps ----
    rows = []
    cell_lu = lumap[cells['cell'].to_numpy()]
    for am, lu in HIR_LU.items():
        j = ag_lus.index(lu)
        lu_area = real_area[lumap == j].sum()
        for yr in YEARS:
            sel = (cell_lu == j) & (cells['fy_registered'].to_numpy() <= yr)
            # overlapping CEAs in one cell are clipped at the whole cell
            per_cell = pd.Series(cells['fraction'].to_numpy()[sel]).groupby(cells['cell'].to_numpy()[sel]).sum().clip(upper=1)
            ha = float((per_cell * real_area[per_cell.index.to_numpy()]).sum()) * scale
            rows.append([am, lu, yr, ha / lu_area, 'CER register + CEA files, mapped area x issuance scale', 'OBSERVED (area) / INFERRED (scale)'])
    for lu in LUS_PA:
        rows.append(['Precision Agriculture', lu, ABS_YEAR, pa, 'ABS 4627.0 2016-17: nitrification-inhibitor + slow-release fertiliser area / crop land',
                     'OBSERVED (upper bound: categories can overlap and include pasture)'])
        rows.append(['AgTech EI', lu, 2010, 0.0, 'none: no observed adoption source for the option (Nick ruling 2)',
                     'ASSUMPTION (observed adoption negligible)'])
    for lu in LUS_BIOCHAR:
        rows.append(['Biochar', lu, ABS_YEAR, bc, 'ABS 4627.0 2016-17: biochar area applied / crop land', 'OBSERVED (RSE 25-50%)'])
    caps = pd.DataFrame(rows, columns=['option', 'land_use', 'year_end', 'cap_fraction', 'source', 'status'])
    caps = caps.sort_values(['option', 'land_use', 'year_end']).reset_index(drop=True)

    write_csv(os.path.join(a.out, 'caps.csv'), caps)
    write_csv(os.path.join(a.out, 'hir_project_cells.csv'), cells)
    write_csv(os.path.join(a.out, 'hir_projects.csv'), proj.drop(columns='cea_url'))
    write_csv(os.path.join(a.out, 'hir_scaling.csv'), scaling)
    write_csv(os.path.join(a.out, 'cea_manifest.csv'), manifest)

    # ---- provenance (last; hashes the other outputs) ----
    me = open(os.path.abspath(__file__), 'rb').read()
    P = [['source', 'CER ACCU Scheme project register (CSV)', CER_REGISTER_URL, sha(reg_raw), 'sha256 of raw file', SOURCE_DATES['CER']],
         ['source', f'CER CEA mapping files ({len(manifest)} HIR projects)', 'cea_manifest.csv', sha(canon),
          'sha256 of sorted project_id,url,sha256 lines', SOURCE_DATES['CER']],
         ['source', 'ABS 4627.0 2016-17, Table 1 Australia', ABS_4627_URL, sha(abs_raw), 'sha256 of raw file', SOURCE_DATES['ABS']],
         ['script', 'luto/tools/ag_mgt_caps.py', '', sha(me), 'sha256 of script', '']]
    for f in sorted(f for f in os.listdir(a.out) if f.endswith('.csv') and f != 'provenance.csv'):
        P.append(['output', f, '', sha(open(os.path.join(a.out, f), 'rb').read()), 'sha256', ''])
    write_csv(os.path.join(a.out, 'provenance.csv'), pd.DataFrame(P, columns=['kind', 'item', 'url_or_endpoint', 'sha256', 'note', 'source_date']))
    print(f'ag_mgt_caps: {len(proj)} HIR projects, {len(manifest)} mapped, scale {scale:.4f}; PA cap {pa:.4f}, Biochar cap {bc:.6f}')


if __name__ == '__main__':
    sys.exit(main())
