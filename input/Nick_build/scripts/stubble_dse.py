#!/usr/bin/env python
"""Build the stubble carrying rate LUTO reads (settings.STUBBLE_DSE_FILE) from Nick's raw rate.

The rule (moved here from luto/data.py `Data.get_stubble_effective_rate`, 2026-11-20): a cell without the sheep per-head
data the stubble code reads carries no stubble sheep, so its rate is 0. Otherwise such a cell gets stubble emissions
(its emission factors exist) with no stubble production, revenue or cost (its economics fields do not). The sheep data
checked, all over the full-resolution land cells:
  - agec_lvstk.nc: the SHEEP fields in SHEEP_AGEC_FIELDS (quantity, revenue, cost, water);
  - agGHG_lvstk.nc: EVERY SHEEP emission source (not only the scope-1 ones LUTO may keep, so the file does not depend on
    settings.USE_GHG_SCOPE_1);
  - water_delivery_price.nc.
A NaN rate (no rate on the cell) becomes 0 too: no stubble sheep there. The file holds no NaN — LUTO reads it as it is
(inputs carry no NaN; a data-processing rule, not a check in LUTO).

The rule is applied at full resolution, before LUTO block-averages the rate at RESFACTOR > 1.

Inputs:  --raw, Nick's raw rate (NetCDF, one float variable over `cell`, every full-resolution land cell, DSE per ha,
         annualised); --input, LUTO's input folder (the three tables above).
Outputs (into --out): stubble_dse_ha.nc (the raw rate's variable, dims and coords, with the rule applied, no NaN) and
         provenance.csv (sha256 of this script, every input and the output; the counts of cells set to 0).

Run as a plain script from the repo root (no luto imports):

    python input/Nick_build/scripts/stubble_dse.py --raw <Nick's raw file> --input input --out input/Nick_build/stubble_dse

then set settings.STUBBLE_DSE_FILE = 'Nick_build/stubble_dse/stubble_dse_ha.nc'.

Third-party packages: numpy, xarray (netCDF4).
"""
from pathlib import Path
import argparse
import csv
import hashlib

import numpy as np
import xarray as xr

# The sheep per-head fields of agec_lvstk.nc the stubble code reads (quantity, revenue, cost, water)
SHEEP_AGEC_FIELDS = ['QC', 'WR_DRN', 'F1', 'Q1', 'P1', 'F2', 'Q2', 'P2', 'F3', 'Q3', 'P3']


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def sheep_columns(table: xr.DataArray, fields=None) -> np.ndarray:
    """The SHEEP columns of a (column, cell) table — those in `fields` (every one when None) — as (n, cell)."""
    animal, field = table['animal'].values, table['field'].values
    picked = (animal == 'SHEEP') & (np.isin(field, fields) if fields is not None else True)
    if fields is not None:
        missing = set(fields) - set(field[animal == 'SHEEP'])
        assert not missing, f'SHEEP fields missing from the table: {sorted(missing)}'
    assert picked.any(), 'no SHEEP column in the table'
    return table.values[np.flatnonzero(picked)]


def get_sheep_data(input_dir: Path) -> np.ndarray:
    """True on every full-resolution cell that has all the sheep per-head data the stubble code reads."""
    with xr.open_dataset(input_dir / 'agec_lvstk.nc') as table:
        economics = sheep_columns(table['AGEC_LVSTK'].load(), SHEEP_AGEC_FIELDS)
    with xr.open_dataset(input_dir / 'agGHG_lvstk.nc') as table:
        emissions = sheep_columns(table['AGGHG_LVSTK'].load())
    water_price = xr.load_dataarray(input_dir / 'water_delivery_price.nc').values
    return (np.isfinite(economics).all(axis=0)
            & np.isfinite(emissions).all(axis=0)
            & np.isfinite(water_price))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--raw', required=True, help="Nick's raw stubble rate (NetCDF, over every full-resolution land cell)")
    parser.add_argument('--input', default='input', help="LUTO's input folder (agec_lvstk.nc, agGHG_lvstk.nc, water_delivery_price.nc)")
    parser.add_argument('--out', required=True, help='output folder, e.g. input/Nick_build/stubble_dse')
    args = parser.parse_args()
    input_dir, out_dir = Path(args.input), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rate = xr.load_dataarray(args.raw)
    assert rate.dims == ('cell',), f'the raw rate must be one variable over cell, not {rate.dims}'
    sheep_data = get_sheep_data(input_dir)
    assert sheep_data.shape == rate.shape, f'the raw rate has {rate.size} cells, the sheep tables {sheep_data.size}'

    has_rate = np.isfinite(rate.values)
    zeroed = has_rate & (rate.values != 0) & ~sheep_data
    effective = rate.where(sheep_data & has_rate, 0.0).astype(rate.dtype)       # no sheep data, or no rate: 0
    assert np.isfinite(effective.values).all(), 'the stubble rate must hold no NaN or inf'
    out_path = out_dir / 'stubble_dse_ha.nc'
    effective.to_netcdf(out_path)
    no_rate = int((~has_rate).sum())
    print(f'{int(zeroed.sum())} cells with a stubble rate have no sheep data: set to 0; {no_rate} cells without a rate: set to 0')

    script = Path(__file__).absolute()
    rows = [['kind', 'item', 'sha256', 'note'],
            ['script', 'input/Nick_build/scripts/stubble_dse.py', sha256(script), 'sha256 of script'],
            ['input', Path(args.raw).name, sha256(args.raw), "Nick's raw stubble rate"]]
    rows += [['input', name, sha256(input_dir / name), 'sheep data checked'] for name in ('agec_lvstk.nc', 'agGHG_lvstk.nc', 'water_delivery_price.nc')]
    rows += [['output', out_path.name, sha256(out_path), f'{int(zeroed.sum())} cells set to 0 for want of sheep data, {no_rate} without a rate set to 0']]
    with open(out_dir / 'provenance.csv', 'w', newline='') as file:
        csv.writer(file, lineterminator='\n').writerows(rows)


if __name__ == '__main__':
    main()
