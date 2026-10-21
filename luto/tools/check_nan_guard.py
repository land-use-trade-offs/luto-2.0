"""
Tests of the NaN guards (`Data.check_eligible_nan`, `Data.check_ag_man_nan`) by injection.

1. `Data` builds on the current inputs: the guard passes.
2. Array classes: one NaN is put in one eligible entry of each guarded array class on the built `Data` (allocated
   entries for the climate multiplier). The guard must fire and name the array, lm, lu, a count of 1, and the
   entry's SA2 and NRM region; after the value is restored it must pass again.
3. Ag-management effects: one NaN in a cost, a revenue and an adoption-cost effect matrix of `get_economics`
   (2010 -> 2020). The guard must fire with the right option, lm and lu, and pass once restored.
4. Fine level (RESFACTOR > 1 only): NaN in the fine `agec_crops` rows of one block's eligible cells for
   (dry, Winter cereals), in memory (pd.read_hdf is patched, `input/` is not written).
   - all of the block's eligible cells: the aggregated entry has no data, so the build must fail with the guard;
   - half of them, the centre cell excluded: the aggregation fills the entry, so the build must pass.

Usage (from the repo root, inputs and other settings from luto/settings.py):
    python -m luto.tools.check_nan_guard [--resfactor N]

Prints PASS / FAIL per case and exits with status 1 if any case fails.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

import luto.settings as settings


FINE_LM, FINE_LU = 'dry', 'Winter cereals'
BEEF, DAIRY = 'Beef - modified land', 'Dairy - modified land'


def report_ok(msg, name, lm, lu, extra=(), single=True) -> bool:
    """True if `msg` is a guard report of one NaN entry in (name, lm, lu), naming every string in `extra`;
    with `single`, that entry is the only one reported."""
    return (msg is not None and (': 1 NaN entries' in msg or not single)
            and re.search(rf'^\s*{re.escape(name)}\s+{lm}\s+{re.escape(lu)}\s+1\s*$', msg, re.M) is not None
            and all(str(e) in msg for e in extra))


def raw_inputs(data):
    """FEED_REQ and WATER_DELIVERY_PRICE before their np.nan_to_num, as `Data.__init__` passes them to the guard."""
    read = lambda f: pd.read_hdf(os.path.join(settings.INPUT_DIR, f), where=data.MASK).to_numpy()
    return read('feed_req.h5'), read('water_delivery_price.h5')


def test_classes(data) -> bool:
    feed_raw, wdp_raw = (np.array(a) for a in raw_inputs(data))       # writable copies
    coarse = getattr(data, 'LVSTK_COARSE', None)
    if coarse is None:                       # RESFACTOR 1: the consumers read the attributes; make them writable float copies
        for name in ('PASTURE_KG_DM_HA', 'SAFE_PUR_NATL', 'SAFE_PUR_MODL'):
            setattr(data, name, np.array(getattr(data, name), dtype=np.float64))   # PASTURE_KG_DM_HA is integer at RF1

    def lvstk(lm, lu, key):                 # the array the consumers read for (lm, lu)
        if coarse is not None:
            return coarse[lm, lu][key]
        return {'AGEC_LVSTK': data.AGEC_LVSTK, 'AGGHG_LVSTK': data.AGGHG_LVSTK, 'AGGHG_IRRPAST': data.AGGHG_IRRPAST,
                'FEED_REQ': feed_raw, 'PASTURE_KG_DM_HA': data.PASTURE_KG_DM_HA,
                'SAFE_PUR': data.SAFE_PUR_MODL if 'modified' in lu else data.SAFE_PUR_NATL,
                'WATER_DELIVERY_PRICE': wdp_raw}[key]

    def entry(m, lu, allocated):            # a cell in the middle of the eligible (or allocated) cells of (m, lu)
        j = data.DESC2AGLU[lu]
        cells = np.nonzero((data.AG_L_MRJ[m, :, j] > 0) if allocated else data.EXCLUDE[m, :, j])[0]
        return int(cells[len(cells) // 2])

    cases = [   # (class, array name in the report, lm, lu, array getter, column or None, allocated entry?)
        ('area', 'AGEC_CROPS AC', 'dry', 'Winter cereals', lambda: data.AGEC_CROPS, ('AC', 'dry', 'Winter cereals'), False),
        ('area (GHG)', 'AGGHG_CROPS CO2E_KG_HA_SOIL', 'dry', 'Winter cereals', lambda: data.AGGHG_CROPS, ('CO2E_KG_HA_SOIL', 'dry', 'Winter cereals'), False),
        ('production', 'AGEC_CROPS QC', 'irr', 'Cotton', lambda: data.AGEC_CROPS, ('QC', 'irr', 'Cotton'), False),
        ('production', 'AGEC_CROPS P1', 'dry', 'Winter cereals', lambda: data.AGEC_CROPS, ('P1', 'dry', 'Winter cereals'), False),
        ('water', 'AGEC_CROPS WP', 'irr', 'Cotton', lambda: data.AGEC_CROPS, ('WP', 'irr', 'Cotton'), False),
        ('area (irrigated pasture GHG)', 'AGGHG_IRRPAST CO2E_KG_HA_IRRIG', 'irr', BEEF, lambda: lvstk('irr', BEEF, 'AGGHG_IRRPAST'), 'CO2E_KG_HA_IRRIG', False),
        ('area (pasture)', 'PASTURE_KG_DM_HA', 'dry', BEEF, lambda: lvstk('dry', BEEF, 'PASTURE_KG_DM_HA'), None, False),
        ('pasture', 'SAFE_PUR_MODL', 'dry', BEEF, lambda: lvstk('dry', BEEF, 'SAFE_PUR'), None, False),
        ('capacity', 'FEED_REQ', 'dry', BEEF, lambda: lvstk('dry', BEEF, 'FEED_REQ'), None, False),
        ('head', 'AGEC_LVSTK QC', 'dry', BEEF, lambda: lvstk('dry', BEEF, 'AGEC_LVSTK'), ('QC', 'BEEF'), False),
        ('head (GHG)', 'AGGHG_LVSTK CO2E_KG_HEAD_ENTERIC', 'dry', BEEF, lambda: lvstk('dry', BEEF, 'AGGHG_LVSTK'), ('BEEF', 'CO2E_KG_HEAD_ENTERIC'), False),
        ('head x F', 'AGEC_LVSTK Q1', 'dry', BEEF, lambda: lvstk('dry', BEEF, 'AGEC_LVSTK'), ('Q1', 'BEEF'), False),
        ('head x F x Q', 'AGEC_LVSTK P1', 'dry', BEEF, lambda: lvstk('dry', BEEF, 'AGEC_LVSTK'), ('P1', 'BEEF'), False),
        ('water (livestock)', 'WATER_DELIVERY_PRICE', 'irr', DAIRY, lambda: lvstk('irr', DAIRY, 'WATER_DELIVERY_PRICE'), None, False),
        ('climate, crop (allocated)', 'CLIMATE_CHANGE_IMPACT (allocated)', 'dry', 'Winter cereals', lambda: data.CLIMATE_CHANGE_IMPACT, ('dry', 'Winter cereals', '2050'), True),
        ('climate, livestock (allocated)', 'CLIMATE_CHANGE_IMPACT (allocated)', 'dry', BEEF, lambda: data.CLIMATE_CHANGE_IMPACT, ('dry', BEEF, '2050'), True),
    ]
    ok_all = True
    for cls, name, lm, lu, get, col, allocated in cases:
        r = entry(data.LANDMANS.index(lm), lu, allocated)
        obj = get()
        if isinstance(obj, pd.DataFrame):
            if col not in obj.columns:      # the climate table's year level is not a string
                col = next(c for c in obj.columns if tuple(map(str, c)) == tuple(map(str, col)))
            ci = obj.columns.get_loc(col)
            old, obj.iat[r, ci] = obj.iat[r, ci], np.nan
        else:
            old, obj[r] = obj[r], np.nan
        try:
            data.check_eligible_nan(feed_raw, wdp_raw)
            msg = None
        except ValueError as e:
            msg = str(e)
        if isinstance(obj, pd.DataFrame):
            obj.iat[r, ci] = old
        else:
            obj[r] = old
        sa2, nrm = int(data.AGGHG_IRRPAST['SA2_ID'].iat[r]), data.REGION_NRM_NAME[r]
        # At RESFACTOR 1 a livestock input is one array per cell (or per animal) shared by several (lm, lu): the NaN is
        # then reported once for each of them that is eligible on the cell. At RESFACTOR > 1 each (lm, lu) has its own copy.
        shared = coarse is None and lu in data.LU_LVSTK
        ok = report_ok(msg, name, lm, lu, (sa2, nrm), single=not shared)
        ok_all &= ok
        print(f"{'PASS' if ok else 'FAIL'} | {cls:30s} | {name:36s} | {lm} | {lu:22s} | cell {r} | SA2 {sa2} | {nrm}", flush=True)
        if not ok:
            print(msg, flush=True)
    data.check_eligible_nan(feed_raw, wdp_raw)
    print('array classes: restored, the guard passes again.', flush=True)
    return ok_all


def test_ag_man(data) -> bool:
    import luto.solvers.row_inputs as row_inputs
    from luto.data import Data

    captured = []
    original = Data.check_ag_man_nan

    def spy(self, effects, yr_cal):
        captured.append({w: {am: a.copy() for am, a in d.items()} for w, d in effects.items()})
        return original(self, effects, yr_cal)

    Data.check_ag_man_nan = spy
    try:
        row_inputs.get_economics(data, data.YR_CAL_BASE, 2020)
    finally:
        Data.check_ag_man_nan = original
    if not captured:
        print('FAIL | ag-management effects | get_economics did not call check_ag_man_nan', flush=True)
        return False
    print(f'ag-management effects: get_economics called check_ag_man_nan {len(captured)} time(s), passed', flush=True)

    effects, ok_all = captured[0], True
    for what, am in [('cost', 'Asparagopsis taxiformis'), ('revenue', 'Precision Agriculture'), ('adoption cost', 'HIR - Beef')]:
        if am not in data.AGMAN2LU or am not in effects.get(what, {}):
            print(f'SKIP | {am} {what} effect (option off)', flush=True)
            continue
        j = data.AGMAN2LU[am][0]
        cells = np.nonzero(data.EXCLUDE[0, :, j])[0]
        r = int(cells[len(cells) // 2])
        arr = effects[what][am]
        old, arr[0, r, 0] = arr[0, r, 0], np.nan
        try:
            data.check_ag_man_nan(effects, 2020)
            msg = None
        except ValueError as e:
            msg = str(e)
        arr[0, r, 0] = old
        name = f'{am} {what} effect (2020)'
        ok = report_ok(msg, name, 'dry', data.AGLU2DESC[j])
        ok_all &= ok
        print(f"{'PASS' if ok else 'FAIL'} | {name} | dry | {data.AGLU2DESC[j]} | cell {r}", flush=True)
        if not ok:
            print(msg, flush=True)
    data.check_ag_man_nan(effects, 2020)
    print('ag-management effects: restored, the guard passes again.', flush=True)
    return ok_all


def build_with_fine_nan(share: str):
    """Build `Data` with NaN in the fine agec_crops rows of one block's eligible cells for (FINE_LM, FINE_LU)
    ('all' of them, or 'half' without the centre cell). Returns the guard message, or None if the build passed."""
    import rasterio
    from luto.data import Data

    rf = settings.RESFACTOR
    with rasterio.open(os.path.join(settings.INPUT_DIR, 'NLUM_2010-11_mask.tif')) as src:
        nlum = src.read(1)
    rows, cols = np.nonzero(nlum)
    lus = pd.read_csv(os.path.join(settings.INPUT_DIR, 'ag_landuses.csv'), header=None)[0].to_list()
    lumask = pd.read_hdf(os.path.join(settings.INPUT_DIR, 'lumap.h5')).to_numpy() != -1
    eligible = np.load(os.path.join(settings.INPUT_DIR, 'x_mrj.npy'))[['dry', 'irr'].index(FINE_LM), :, lus.index(FINE_LU)] & lumask
    key = (rows // rf) * -(-nlum.shape[1] // rf) + cols // rf
    is_centre = ((rows % rf) == rf // 2) & ((cols % rf) == rf // 2)
    count = pd.Series(eligible).groupby(key).sum()
    centre_eligible = pd.Series(eligible & is_centre).groupby(key).any()
    candidates = [k for k in count[count >= rf * rf // 2].index if centre_eligible[k]]
    block = candidates[len(candidates) // 2]
    cells = np.nonzero((key == block) & eligible)[0]
    if share == 'all':
        inject = cells
    else:
        inject = cells[~is_centre[cells]][: len(cells) // 2]
    print(f'fine level ({share}): block {block}, {len(cells)} eligible fine cells for ({FINE_LM}, {FINE_LU}); '
          f'NaN in {len(inject)} (centre cell included: {bool(is_centre[inject].any())})', flush=True)

    read_hdf = pd.read_hdf

    def patched(path, *a, **k):
        df = read_hdf(path, *a, **k)
        if not str(path).endswith('agec_crops.h5'):
            return df
        if 'where' in k:                                    # the centre-cell read
            full = read_hdf(path).copy()
            cols_ = full.columns.get_indexer([c for c in full.columns if c[1:] == (FINE_LM, FINE_LU)])
            full.iloc[inject, cols_] = np.nan
            return full[np.asarray(k['where'])]
        s0 = k.get('start') or 0                            # a band read (start / stop) or a whole-table read
        loc = inject[(inject >= s0) & (inject < s0 + len(df))] - s0
        if loc.size:
            df = df.copy()
            df.iloc[loc, df.columns.get_indexer([c for c in df.columns if c[1:] == (FINE_LM, FINE_LU)])] = np.nan
        return df

    pd.read_hdf = patched
    try:
        Data()
        return None
    except ValueError as e:
        return str(e)
    finally:
        pd.read_hdf = read_hdf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--resfactor', type=int, default=None, help='override settings.RESFACTOR')
    args = parser.parse_args()
    if args.resfactor is not None:
        settings.RESFACTOR = args.resfactor
    print(f'NaN guard tests at RESFACTOR {settings.RESFACTOR}', flush=True)

    from luto.data import Data
    try:
        data = Data()
    except ValueError as e:
        print('FAIL | current inputs | the guard fired on the current inputs:\n' + str(e), flush=True)
        return 1
    print('PASS | current inputs | Data built, the guard passed', flush=True)

    results = {'array classes': test_classes(data), 'ag-management effects': test_ag_man(data)}
    del data

    if settings.RESFACTOR > 1:
        msg = build_with_fine_nan('all')
        ok = msg is not None and report_ok(msg, 'AGEC_CROPS Yield', FINE_LM, FINE_LU, single=False)
        print(f"{'PASS' if ok else 'FAIL'} | fine level, all eligible cells NaN | the guard "
              f"{'fired' if msg else 'did not fire'}", flush=True)
        if msg:
            print(msg, flush=True)
        results['fine level, all'] = ok
        msg = build_with_fine_nan('half')
        ok = msg is None
        print(f"{'PASS' if ok else 'FAIL'} | fine level, half the eligible cells NaN | the build "
              f"{'passed' if ok else 'failed'}", flush=True)
        if msg:
            print(msg, flush=True)
        results['fine level, half'] = ok

    print('\n' + '\n'.join(f"{'PASS' if v else 'FAIL'} | {k}" for k, v in results.items()), flush=True)
    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
