"""
Nesting check for the coarse aggregation at RESFACTOR > 1.

For every (lm, lu) (production: every (lm, product)) and every eligible coarse cell (block), the coarse value in the
2010 agricultural matrices, scaled to the block's eligible area, equals the total of the RESFACTOR 1 values over the
block's fine cells that are eligible at RESFACTOR 1:

    fine total      T_b = sum_i M_1[m, i, j]                    i: fine cells of block b eligible for (m, j)
    coarse product  P_b = M_rf[m, b, j] / REAL_AREA_b * A_b      A_b: sum of those cells' REAL_AREA_NO_RESFACTOR

M is a per-cell value (per ha x REAL_AREA), so M_rf / REAL_AREA is the coarse per-ha value. The check reads the cost,
revenue, GHG, water requirement and production matrices for 2010, when the climate multiplier is 1. DYNAMIC_PRICE is
switched off for the builds because the 2010 price multiplier is not exactly 1 at RESFACTOR > 1 (a separate issue).

2050 production is reported as a diagnostic only: the weights are year-invariant, so livestock production (head x C x F
x Q, with F and Q weighted by head) nests only where the climate multiplier does not co-vary with F x Q in a block.

A product made by several land uses is checked over the cells eligible for any of them. With LVSTK_K_FILE and
STUBBLE_DSE_FILE set in settings.py the check covers per-type k and the stubble products (one set per stubble land use);
the stubble rows (the stubble products, and the cost, revenue, GHG and water of the stubble land uses, which add the
stubble sheep to the crop) are also reported on their own, with national and worst-block residuals.

Usage (from the repo root, inputs and other settings from luto/settings.py):
    python -m luto.tools.check_nesting 10 20 [--out DIR] [--tol 1e-4]

Writes nesting_rf<RF>.csv (one row per quantity, lm and lu or product) to DIR (default: the current directory),
prints a summary per quantity, and exits with status 1 if any 2010 block has a relative error above `tol`.
"""

import argparse
import gc
import os
import sys

import numpy as np
import pandas as pd

import luto.settings as settings


QUANTITIES = ['cost', 'revenue', 'GHG', 'water requirement', 'production', 'production 2050 (diagnostic)']


def build_matrices(resfactor: int) -> dict:
    """Build `Data` at `resfactor` and return the 2010 (and 2050 production) matrices and the masks the check needs."""
    settings.RESFACTOR = resfactor
    settings.DYNAMIC_PRICE = False

    from luto.data import Data
    import luto.economics.agricultural.cost as ag_cost
    import luto.economics.agricultural.revenue as ag_revenue
    import luto.economics.agricultural.quantity as ag_quantity
    import luto.economics.agricultural.ghg as ag_ghg
    import luto.economics.agricultural.water as ag_water

    data = Data()
    out = {
        'cost': np.asarray(ag_cost.get_cost_matrices(data, 0)),
        'revenue': np.asarray(ag_revenue.get_rev_matrices(data, 0)),
        'GHG': np.asarray(ag_ghg.get_ghg_matrices(data, 0)),
        'water requirement': np.asarray(ag_water.get_wreq_matrices(data, 0)),
        'production': np.asarray(ag_quantity.get_quantity_matrices(data, 0)),
        'production 2050 (diagnostic)': np.asarray(ag_quantity.get_quantity_matrices(data, 2050 - data.YR_CAL_BASE)),
        'exclude': np.asarray(data.EXCLUDE).astype(bool),
        'real_area': np.asarray(data.REAL_AREA, dtype=np.float64),
        'lu2pr': np.asarray(data.LU2PR),
        'mask': np.asarray(data.MASK),
        'landmans': list(data.LANDMANS),
        'landuses': list(data.AGRICULTURAL_LANDUSES),
        'stubble_landuses': list(getattr(data, 'LU_STUBBLE', [])),
        'products': list(data.PRODUCTS),
    }
    if resfactor == 1:
        out.update(lumask=np.asarray(data.LUMASK), fine_area=np.asarray(data.REAL_AREA_NO_RESFACTOR, dtype=np.float64),
                   nlum_mask=np.asarray(data.NLUM_MASK))
    del data
    gc.collect()
    return out


def block_of_fine_cells(nlum_mask: np.ndarray, mask_rf: np.ndarray, rf: int) -> np.ndarray:
    """Coarse cell index of every fine land cell (-1 where its block holds no coarse cell), from the row-major
    block keys of the centre cells (`mask_rf` marks them among the fine land cells)."""
    rows, cols = np.nonzero(nlum_mask)
    n_block_cols = -(-nlum_mask.shape[1] // rf)
    fine_key = (rows // rf) * n_block_cols + cols // rf
    coarse_key = fine_key[mask_rf]
    assert coarse_key.size == 0 or np.all(np.diff(coarse_key) > 0), 'coarse cells are not in row-major block order'
    pos = np.minimum(np.searchsorted(coarse_key, fine_key), coarse_key.size - 1)
    return np.where(coarse_key[pos] == fine_key, pos, -1)


def compare(fine: dict, coarse: dict, rf: int) -> pd.DataFrame:
    """One row per (quantity, lm, lu or product): blocks, fine and coarse totals, maximum relative error."""
    f2c = block_of_fine_cells(fine['nlum_mask'], coarse['mask'], rf)[fine['lumask']]   # RF1 cell -> coarse cell
    n_coarse = coarse['real_area'].size

    def per_block(values, eligible):
        k = eligible & (f2c >= 0)
        return np.bincount(f2c[k], weights=values[k].astype(np.float64), minlength=n_coarse)

    rows = []
    for qty in QUANTITIES:
        is_prod = qty.startswith('production')
        names = fine['products'] if is_prod else fine['landuses']
        for m, lm in enumerate(fine['landmans']):
            for k, name in enumerate(names):
                M1, Mn = fine[qty][m, :, k], coarse[qty][m, :, k]
                if not (M1 != 0).any() and not (Mn != 0).any():
                    continue
                js = np.nonzero(fine['lu2pr'][k])[0] if is_prod else [k]           # LU2PR is (product, lu)
                e1, en = fine['exclude'][m][:, js].any(axis=1), coarse['exclude'][m][:, js].any(axis=1)
                if not en.any():
                    continue
                T = per_block(M1, e1)
                P = Mn.astype(np.float64) / coarse['real_area'] * per_block(fine['real_area'], e1)
                d = np.abs(P[en] - T[en])
                s = np.maximum(np.abs(T[en]), np.abs(P[en]))
                rel = np.divide(d, s, out=np.zeros_like(d), where=s > 0)
                fine_total, coarse_total = float(T[en].sum()), float(np.nansum(P[en]))
                stubble = ('STUBBLE' in name) if is_prod else (name in fine['stubble_landuses'])
                rows.append(dict(quantity=qty, lm=lm, key=name, stubble=stubble, blocks=int(en.sum()), fine_total=fine_total,
                                 coarse_total=coarse_total, max_rel_error=float(np.nanmax(rel)),
                                 max_abs_error=float(np.nanmax(d)), abs_at_max_rel=float(d[np.nanargmax(rel)]),
                                 national_abs=abs(coarse_total - fine_total),
                                 national_rel=abs(coarse_total - fine_total) / abs(fine_total) if fine_total else 0.0,
                                 blocks_above_tol=0, nan_blocks=int(np.isnan(P[en]).sum()), _rel=rel))
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('resfactors', type=int, nargs='+', help='coarse resolutions to check, each > 1')
    parser.add_argument('--out', default='.', help='folder for nesting_rf<RF>.csv')
    parser.add_argument('--tol', type=float, default=1e-4, help='relative error above which a 2010 block fails')
    args = parser.parse_args()
    assert all(rf > 1 for rf in args.resfactors), 'RESFACTOR 1 is the reference'

    fine = build_matrices(1)
    failed = False
    for rf in args.resfactors:
        coarse = build_matrices(rf)
        df = compare(fine, coarse, rf)
        df['blocks_above_tol'] = [int((r > args.tol).sum()) for r in df.pop('_rel')]
        df.to_csv(os.path.join(args.out, f'nesting_rf{rf}.csv'), index=False)

        summary = df.groupby('quantity', sort=False).agg(
            entries=('key', 'size'), blocks=('blocks', 'sum'), fine_total=('fine_total', 'sum'),
            coarse_total=('coarse_total', 'sum'), max_rel_error=('max_rel_error', 'max'),
            blocks_above_tol=('blocks_above_tol', 'sum'), nan_blocks=('nan_blocks', 'sum'))
        print(f'\nRESFACTOR {rf} (tolerance {args.tol:g} relative, per block)')
        print(summary.to_string(float_format=lambda v: f'{v:.6e}'))
        bad = summary.drop(index=[q for q in summary.index if 'diagnostic' in q])
        rf_failed = bool((bad['blocks_above_tol'] > 0).any() or (bad['nan_blocks'] > 0).any())
        print(f'RESFACTOR {rf}: {"FAIL" if rf_failed else "PASS"} (2010)', flush=True)
        if df['stubble'].any():
            st = df[df['stubble']]
            print(f'RESFACTOR {rf}, stubble rows (stubble products; cost, revenue, GHG, water of the stubble land uses):')
            print(st[['quantity', 'lm', 'key', 'blocks', 'fine_total', 'coarse_total', 'national_abs', 'national_rel',
                      'max_abs_error', 'max_rel_error', 'abs_at_max_rel', 'blocks_above_tol']]
                  .to_string(index=False, float_format=lambda v: f'{v:.6e}'), flush=True)
        failed |= rf_failed
        del coarse
        gc.collect()
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
