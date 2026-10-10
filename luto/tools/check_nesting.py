"""
Nesting and base-year fidelity checks for the coarse aggregation at RESFACTOR > 1.

Nesting. For every (lm, lu) (production: every (lm, product)) and every eligible coarse cell (block), the coarse value
in the 2010 agricultural matrices, scaled to the area of the block's cell set, equals the total of the RESFACTOR 1
values over the same fine cells:

    fine total      T_b = sum_i M_1[m, i, j]                    i: the cell set of (m, j) in block b
    coarse product  P_b = M_rf[m, b, j] / REAL_AREA_b * A_b      A_b: sum of those cells' REAL_AREA_NO_RESFACTOR

The cell set is the one `Data.coarsen_guarded_arrays` averages over: the block's fine cells allocated to (m, j) in the
base year where the block has any ('allocated'), else the block's fine cells eligible at RESFACTOR 1 ('eligible').
M is a per-cell value (per ha x REAL_AREA), so M_rf / REAL_AREA is the coarse per-ha value. The check reads the cost,
revenue, GHG, water requirement and production matrices for 2010, when the climate multiplier is 1. DYNAMIC_PRICE is
switched off for the builds because the 2010 price multiplier is not exactly 1 at RESFACTOR > 1 (a separate issue).

Diagnostics, reported and not gated:
- 2050 production: the weights are year-invariant, so livestock production (head x C x F x Q, with F and Q weighted by
  head) nests only where the climate multiplier does not co-vary with F x Q in a block.
- '<quantity> vs eligible mean': for allocated entries, the coarse value scaled to the block's eligible area against
  the RESFACTOR 1 total over the eligible cells, the value land-use change into the rest of the block sees.

Each product is made by one land use (asserted). With LVSTK_K_FILE and STUBBLE_DSE_FILE set in settings.py the check
covers per-type k and the stubble products (one set per stubble land use); the stubble rows are also reported on their
own, with national and worst-block residuals.

Base-year fidelity. The base-year totals at RESFACTOR > 1 (AG_L_MRJ x the coarse matrices) against RESFACTOR 1
(AG_L_MRJ x the RESFACTOR 1 matrices), by state and nationally: head by type (beef, sheep, dairy, stubble sheep),
production by product and GHG by land-use group (beef, sheep, dairy, crops). Base-year totals come from allocated
entries only. A coarse cell reports in its centre cell's state, so the RESFACTOR 1 cells are compared in their block's
state; two rows are reported apart and not gated:
- 'lost blocks': RESFACTOR 1 cells whose block has no coarse cell (its centre cell is not land);
- 'state assignment': the RESFACTOR 1 totals in each cell's own state less the same in its block's state.
- 'REAL_AREA basis': AG_L_MRJ x REAL_AREA is the allocated count x the centre cell's area, not the allocated cells'
  own area; the term is the coarse per-ha value x that difference, computed directly, by state and nationally, with
  its residual against (raw - net).
Gated: head and GHG on the raw totals; production net of the REAL_AREA basis term (the coarse per-ha value x the
allocated cells' area); national |RF - RF1| <= 1e-4 x |RF1|; by state |RF - RF1| <= max(5e-4 x |RF1|, 10) (head, t,
or kl for milk). Base-year cost and water requirement are reported, raw and net, not gated.

Usage (from the repo root, inputs and other settings from luto/settings.py):
    python -m luto.tools.check_nesting 10 20 [--out DIR] [--tol 1e-4]

Writes nesting_rf<RF>.csv (one row per quantity, lm, lu or product and cell set) and fidelity_rf<RF>.csv to DIR
(default: the current directory), prints a summary of each, and exits with status 1 if any 2010 block has a relative
error above `tol` or a base-year fidelity row is outside its tolerance.
"""

import argparse
import gc
import os
import sys

import numpy as np
import pandas as pd

import luto.settings as settings


QUANTITIES = ['cost', 'revenue', 'GHG', 'water requirement', 'production', 'production 2050 (diagnostic)']
FIDELITY_NATIONAL_TOL = 1e-4
FIDELITY_STATE_TOL, FIDELITY_STATE_FLOOR = 5e-4, 10.0


def build_matrices(resfactor: int) -> dict:
    """Build `Data` at `resfactor` and return the 2010 (and 2050 production) matrices, the base-year allocation, head
    and the masks the checks need."""
    settings.RESFACTOR = resfactor
    settings.DYNAMIC_PRICE = False

    from luto.data import Data
    import luto.economics.agricultural.cost as ag_cost
    import luto.economics.agricultural.revenue as ag_revenue
    import luto.economics.agricultural.quantity as ag_quantity
    import luto.economics.agricultural.ghg as ag_ghg
    import luto.economics.agricultural.water as ag_water

    data = Data()
    real_area = np.asarray(data.REAL_AREA, dtype=np.float64)
    head = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))
    stubble_head = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))
    for m, lm in enumerate(data.LANDMANS):
        for lu in data.LU_LVSTK:
            animal, vegtype = ag_quantity.lvs_veg_types(lu)
            head[m, :, data.DESC2AGLU[lu]] = np.nan_to_num(ag_quantity.get_yield_pot(data, animal, vegtype, lm, 0)) * real_area
        for lu in getattr(data, 'LU_STUBBLE', []):
            stubble_head[m, :, data.DESC2AGLU[lu]] = np.nan_to_num(ag_quantity.get_stubble_yield_pot(data, lu, lm)) * real_area
    out = {
        'cost': np.asarray(ag_cost.get_cost_matrices(data, 0)),
        'revenue': np.asarray(ag_revenue.get_rev_matrices(data, 0)),
        'GHG': np.asarray(ag_ghg.get_ghg_matrices(data, 0)),
        'water requirement': np.asarray(ag_water.get_wreq_matrices(data, 0)),
        'production': np.asarray(ag_quantity.get_quantity_matrices(data, 0)),
        'production 2050 (diagnostic)': np.asarray(ag_quantity.get_quantity_matrices(data, 2050 - data.YR_CAL_BASE)),
        'head': head,
        'stubble head': stubble_head,
        'base_dvars': np.asarray(data.ag_dvars[data.YR_CAL_BASE], dtype=np.float64),
        'state': np.asarray(data.REGION_STATE_NAME).astype(str),
        'exclude': np.asarray(data.EXCLUDE).astype(bool),
        'real_area': real_area,
        'lu2pr': np.asarray(data.LU2PR),
        'mask': np.asarray(data.MASK),
        'landmans': list(data.LANDMANS),
        'landuses': list(data.AGRICULTURAL_LANDUSES),
        'lu_lvstk': list(data.LU_LVSTK),
        'stubble_landuses': list(getattr(data, 'LU_STUBBLE', [])),
        'products': list(data.PRODUCTS),
    }
    if resfactor == 1:
        lumask = np.asarray(data.LUMASK)
        out.update(lumask=lumask, fine_area=np.asarray(data.REAL_AREA_NO_RESFACTOR, dtype=np.float64),
                   nlum_mask=np.asarray(data.NLUM_MASK),
                   lumap=np.asarray(data.LUMAP_NO_RESFACTOR)[lumask], lmmap=np.asarray(data.LMMAP_NO_RESFACTOR)[lumask])
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
    """One row per (quantity, lm, lu or product, cell set): blocks, fine and coarse totals, maximum relative error."""
    f2c = block_of_fine_cells(fine['nlum_mask'], coarse['mask'], rf)[fine['lumask']]   # RF1 cell -> coarse cell
    n_coarse = coarse['real_area'].size

    def per_block(values, cells):
        k = cells & (f2c >= 0)
        return np.bincount(f2c[k], weights=values[k].astype(np.float64), minlength=n_coarse)

    def cell_set(m, j, e1):
        # The cells coarsen_guarded_arrays averages (m, j) over: allocated where the block has any, else eligible.
        alloc = (fine['lumap'] == j) & (fine['lmmap'] == m) & e1
        has_alloc = per_block(alloc.astype(np.float64), alloc) > 0
        return np.where(has_alloc[np.maximum(f2c, 0)], alloc, e1), has_alloc

    def residuals(M1, Mn, cells, blocks):
        T = per_block(M1, cells)
        P = Mn.astype(np.float64) / coarse['real_area'] * per_block(fine['real_area'], cells)
        d = np.abs(P[blocks] - T[blocks])
        s = np.maximum(np.abs(T[blocks]), np.abs(P[blocks]))
        rel = np.divide(d, s, out=np.zeros_like(d), where=s > 0)
        return T[blocks], P[blocks], d, rel

    lu2pr = fine['lu2pr']                                                              # (product, lu)
    assert (lu2pr.sum(axis=1) == 1).all(), 'a product is made by more than one land use'
    rows = []
    for qty in QUANTITIES:
        is_prod = qty.startswith('production')
        names = fine['products'] if is_prod else fine['landuses']
        for m, lm in enumerate(fine['landmans']):
            for k, name in enumerate(names):
                M1, Mn = fine[qty][m, :, k], coarse[qty][m, :, k]
                if not (M1 != 0).any() and not (Mn != 0).any():
                    continue
                j = int(np.nonzero(lu2pr[k])[0][0]) if is_prod else k
                e1, en = fine['exclude'][m][:, j], coarse['exclude'][m][:, j]
                if not en.any():
                    continue
                cells, has_alloc = cell_set(m, j, e1)
                stubble = ('STUBBLE' in name) if is_prod else (name in fine['stubble_landuses'])
                groups = [('allocated', en & has_alloc, qty, cells), ('eligible', en & ~has_alloc, qty, cells)]
                if 'diagnostic' not in qty:
                    groups.append(('allocated', en & has_alloc, f'{qty} vs eligible mean (diagnostic)', e1))
                for cell_set_name, blocks, qname, cs in groups:
                    if not blocks.any():
                        continue
                    T, P, d, rel = residuals(M1, Mn, cs, blocks)
                    fine_total, coarse_total = float(T.sum()), float(np.nansum(P))
                    rows.append(dict(quantity=qname, lm=lm, key=name, cell_set=cell_set_name, stubble=stubble,
                                     blocks=int(blocks.sum()), fine_total=fine_total, coarse_total=coarse_total,
                                     max_rel_error=float(np.nanmax(rel)), max_abs_error=float(np.nanmax(d)),
                                     abs_at_max_rel=float(d[np.nanargmax(rel)]), national_abs=abs(coarse_total - fine_total),
                                     national_rel=abs(coarse_total - fine_total) / abs(fine_total) if fine_total else 0.0,
                                     blocks_above_tol=0, nan_blocks=int(np.isnan(P).sum()), _rel=rel))
    return pd.DataFrame(rows)


def base_year_totals(mats: dict, alloc_area: np.ndarray = None) -> dict:
    """{(quantity, group): (raw, net)}, per-cell base-year values summed over lm and the group's land uses.
    raw: AG_L_MRJ x the matrices. net (RESFACTOR > 1, given `alloc_area`, the (lm, cell, lu) area of the allocated fine
    cells): the coarse per-ha value (matrix / REAL_AREA) x that area, i.e. the total without the REAL_AREA basis term
    (allocated count x the centre cell's area against the allocated cells' own areas). At RESFACTOR 1 net = raw."""
    X, RA = mats['base_dvars'], mats['real_area']
    landuses, products = mats['landuses'], mats['products']
    out = {}

    def add(key, M, j):
        raw = (X[:, :, j] * M).sum(axis=0)
        net = raw if alloc_area is None else (M / RA * alloc_area[:, :, j]).sum(axis=0)
        r0, n0 = out.get(key, (0, 0))
        out[key] = (r0 + raw, n0 + net)

    for lu in mats['lu_lvstk']:
        j = landuses.index(lu)
        animal = lu.split(' - ')[0].upper()
        add(('head', animal), mats['head'][:, :, j], j)
        add(('GHG t', animal), mats['GHG'][:, :, j], j)
        add(('cost', animal), mats['cost'][:, :, j], j)
        add(('water requirement', animal), mats['water requirement'][:, :, j], j)
    for lu in mats['stubble_landuses']:
        j = landuses.index(lu)
        add(('head', 'SHEEP stubble'), mats['stubble head'][:, :, j], j)
    for lu in landuses:
        if lu not in mats['lu_lvstk'] and not lu.startswith('Unallocated'):
            j = landuses.index(lu)
            add(('GHG t', 'CROPS'), mats['GHG'][:, :, j], j)
            add(('cost', 'CROPS'), mats['cost'][:, :, j], j)
            add(('water requirement', 'CROPS'), mats['water requirement'][:, :, j], j)
    for k, pr in enumerate(products):
        j = int(np.nonzero(mats['lu2pr'][k])[0][0])
        add(('production', pr), mats['production'][:, :, k], j)
    return out


GATED_RAW = ('head', 'GHG t')           # gated on the raw totals
GATED_NET = ('production',)             # gated net of the REAL_AREA basis term (R-365)


def fidelity(fine: dict, coarse: dict, rf: int) -> pd.DataFrame:
    """One row per (quantity, group, state) with 'AUS', and rows apart, not gated: 'lost blocks', 'state assignment <state>'
    and 'REAL_AREA basis' / 'REAL_AREA basis <state>'. Head and GHG are gated on raw totals, production net of the
    REAL_AREA basis term; cost and water requirement are reported, not gated."""
    f2c = block_of_fine_cells(fine['nlum_mask'], coarse['mask'], rf)[fine['lumask']]
    kept = f2c >= 0
    st_block = np.where(kept, coarse['state'][np.maximum(f2c, 0)], '')
    st_own = fine['state']
    n_lm, n_lu, n_coarse = len(fine['landmans']), len(fine['landuses']), coarse['real_area'].size
    alloc_area = np.zeros((n_lm, n_coarse, n_lu))
    for m in range(n_lm):
        for j in range(n_lu):
            k = kept & (fine['lumap'] == j) & (fine['lmmap'] == m)
            if k.any():
                alloc_area[m, :, j] = np.bincount(f2c[k], weights=fine['real_area'][k], minlength=n_coarse)
    T1, Tn = base_year_totals(fine), base_year_totals(coarse, alloc_area)
    rows = []
    for key in T1:
        v1 = np.asarray(T1[key][0], dtype=np.float64)
        raw, net = (np.asarray(x, dtype=np.float64) for x in Tn[key])
        gated, use_net = key[0] in GATED_RAW + GATED_NET, key[0] in GATED_NET
        rf1 = pd.Series(v1[kept]).groupby(st_block[kept]).sum()
        s_raw, s_net = pd.Series(raw).groupby(coarse['state']).sum(), pd.Series(net).groupby(coarse['state']).sum()
        own = pd.Series(v1).groupby(st_own).sum()
        states = sorted(set(rf1.index) | set(s_raw.index) | set(own.index))

        def at(series, s):
            return float(series.sum()) if s == 'AUS' else float(series.get(s, 0.0))

        for s in states + ['AUS']:
            a, r, n = at(rf1, s), at(s_raw, s), at(s_net, s)
            b = n if use_net else r
            tol = FIDELITY_NATIONAL_TOL * abs(a) if s == 'AUS' else max(FIDELITY_STATE_TOL * abs(a), FIDELITY_STATE_FLOOR)
            rows.append(dict(quantity=key[0], group=key[1], state=s, basis='net' if use_net else 'raw', rf1=a, rf=b,
                             diff=b - a, rel=(b - a) / a if a else 0.0, raw_diff=r - a, net_diff=n - a,
                             tol=tol if gated else np.nan, gated=gated, ok=(abs(b - a) <= tol) if gated else True))
        # The REAL_AREA basis term on its own: sum over lm of matrix / REAL_AREA x (AG_L_MRJ x REAL_AREA - allocated area)
        basis = basis_term(coarse, alloc_area, key)
        s_basis = pd.Series(basis).groupby(coarse['state']).sum()
        for s in states + ['AUS']:
            v = at(s_basis, s)
            rows.append(dict(quantity=key[0], group=key[1], state='REAL_AREA basis' + ('' if s == 'AUS' else f' {s}'),
                             basis='', rf1=np.nan, rf=np.nan, diff=v, rel=np.nan, raw_diff=np.nan, net_diff=np.nan,
                             residual=v - (at(s_raw, s) - at(s_net, s)), tol=np.nan, gated=False, ok=True))
        rows.append(dict(quantity=key[0], group=key[1], state='lost blocks', basis='', rf1=float(v1[~kept].sum()), rf=0.0,
                         diff=-float(v1[~kept].sum()), rel=np.nan, tol=np.nan, gated=False, ok=True))
        centre_state = pd.Series(v1).groupby(np.where(kept, st_block, st_own)).sum()
        for s in states:
            rows.append(dict(quantity=key[0], group=key[1], state=f'state assignment {s}', basis='', rf1=np.nan, rf=np.nan,
                             diff=float(own.get(s, 0.0)) - float(centre_state.get(s, 0.0)), rel=np.nan, tol=np.nan,
                             gated=False, ok=True))
    return pd.DataFrame(rows)


def basis_term(mats: dict, alloc_area: np.ndarray, key: tuple) -> np.ndarray:
    """The REAL_AREA basis term of `key` per coarse cell, computed directly: sum over lm and the group's land uses of
    matrix / REAL_AREA x (AG_L_MRJ x REAL_AREA - the allocated cells' area)."""
    X, RA = mats['base_dvars'], mats['real_area']
    landuses = mats['landuses']
    q, g = key
    mat = {'head': 'head', 'GHG t': 'GHG', 'cost': 'cost', 'water requirement': 'water requirement'}.get(q, q)
    if q == 'production':
        k = mats['products'].index(g)
        pairs = [(mats['production'][:, :, k], int(np.nonzero(mats['lu2pr'][k])[0][0]))]
    elif q == 'head' and g == 'SHEEP stubble':
        pairs = [(mats['stubble head'][:, :, landuses.index(lu)], landuses.index(lu)) for lu in mats['stubble_landuses']]
    elif g == 'CROPS':
        pairs = [(mats[mat][:, :, j], j) for j, lu in enumerate(landuses) if lu not in mats['lu_lvstk'] and not lu.startswith('Unallocated')]
    else:
        lus = [lu for lu in mats['lu_lvstk'] if lu.split(' - ')[0].upper() == g]
        pairs = [(mats[mat][:, :, landuses.index(lu)], landuses.index(lu)) for lu in lus]
    out = np.zeros(RA.size)
    for M, j in pairs:
        out += (M / RA * (X[:, :, j] * RA - alloc_area[:, :, j])).sum(axis=0)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('resfactors', type=int, nargs='+', help='coarse resolutions to check, each > 1')
    parser.add_argument('--out', default='.', help='folder for nesting_rf<RF>.csv and fidelity_rf<RF>.csv')
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

        summary = df.groupby(['quantity', 'cell_set'], sort=False).agg(
            entries=('key', 'size'), blocks=('blocks', 'sum'), fine_total=('fine_total', 'sum'),
            coarse_total=('coarse_total', 'sum'), max_rel_error=('max_rel_error', 'max'),
            blocks_above_tol=('blocks_above_tol', 'sum'), nan_blocks=('nan_blocks', 'sum'))
        print(f'\nRESFACTOR {rf} (tolerance {args.tol:g} relative, per block)')
        print(summary.to_string(float_format=lambda v: f'{v:.6e}'))
        bad = summary.drop(index=[q for q in summary.index if 'diagnostic' in q[0]])
        rf_failed = bool((bad['blocks_above_tol'] > 0).any() or (bad['nan_blocks'] > 0).any())
        print(f'RESFACTOR {rf}: nesting {"FAIL" if rf_failed else "PASS"} (2010)', flush=True)
        if df['stubble'].any():
            st = df[df['stubble'] & ~df['quantity'].str.contains('vs eligible mean')]
            print(f'RESFACTOR {rf}, stubble rows (stubble products; cost, revenue, GHG, water of the stubble land uses):')
            print(st[['quantity', 'lm', 'key', 'cell_set', 'blocks', 'fine_total', 'coarse_total', 'national_abs',
                      'national_rel', 'max_abs_error', 'max_rel_error', 'abs_at_max_rel', 'blocks_above_tol']]
                  .to_string(index=False, float_format=lambda v: f'{v:.6e}'), flush=True)

        fd = fidelity(fine, coarse, rf)
        fd.to_csv(os.path.join(args.out, f'fidelity_rf{rf}.csv'), index=False)
        gated = fd[fd['gated']]
        nat = gated[gated['state'] == 'AUS']
        print(f'\nRESFACTOR {rf}: base-year fidelity, national (tolerance {FIDELITY_NATIONAL_TOL:g} relative):')
        print(nat[['quantity', 'group', 'basis', 'rf1', 'rf', 'diff', 'rel', 'raw_diff', 'ok']]
              .to_string(index=False, float_format=lambda v: f'{v:.6e}'))
        ra = fd[fd['state'] == 'REAL_AREA basis']
        print(f'RESFACTOR {rf}: REAL_AREA basis (not gated; computed directly, against raw - net the largest |residual| '
              f'over all rows is {fd["residual"].abs().max():.3e}):')
        print(ra[['quantity', 'group', 'diff', 'residual']].to_string(index=False, float_format=lambda v: f'{v:.6e}'))
        lost = fd[fd['state'] == 'lost blocks']
        print(f'RESFACTOR {rf}: lost blocks (not gated):')
        print(lost[['quantity', 'group', 'rf1']].to_string(index=False, float_format=lambda v: f'{v:.6e}'))
        st_rows = gated[gated['state'] != 'AUS']
        worst = st_rows.loc[(st_rows['diff'].abs() / st_rows['tol']).idxmax()]
        print(f'RESFACTOR {rf}: by state, {int(st_rows["ok"].sum())} of {len(st_rows)} rows within '
              f'max({FIDELITY_STATE_TOL:g} relative, {FIDELITY_STATE_FLOOR:g}); worst {worst["quantity"]} {worst["group"]} '
              f'{worst["state"]}: {worst["diff"]:.6g} against {worst["tol"]:.6g}')
        if not gated['ok'].all():
            print(gated[~gated['ok']].to_string(index=False, float_format=lambda v: f'{v:.6e}'))
        fid_failed = not bool(gated['ok'].all())
        print(f'RESFACTOR {rf}: base-year fidelity {"FAIL" if fid_failed else "PASS"}', flush=True)
        failed |= rf_failed or fid_failed
        del coarse
        gc.collect()
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
