# 2021 base year: status, design and next steps

A short note for agreeing the next steps before anything is rebuilt. Every figure states where it was measured and on what
code. "RES" is `RESFACTOR`.

## Questions for you

1. **The base-year design** ([section 2](#2-base-year-design)): are you happy to move to 2020-21 on the NLUM v7.1 map, with
   livestock placed against the inventory and the two emission rows, before anything is rebuilt?
2. **Where the build tools live** ([section 6](#6-proposals-for-your-decision)): a separate repository writing NetCDF into
   `input/Nick_build/`, with LUTO only reading the files. Does that work for you, or would you prefer something else?
3. **The shape of the remaining work** ([section 6](#6-proposals-for-your-decision)): the proposed order of one-change PRs, and
   whether the coarse-resolution fixes should be tests for your refactor rather than PRs.

## 1. Status

**In `jinzhu`.** Every PR we had open is merged: #227 (NaN guard), #228 (coarse stream length), #229 (two GHG rows), #230
(per-type stocking calibration and cereal stubble), #231 (coarse cells averaged over base-year cells), #215, #217 and #219. The
model code of each is in, ported to the xarray reads. Only our validation scripts (`check_nesting.py`, `check_nan_guard.py`)
were taken out, as `COLLABORATION.md` asks.

**Not published, held until we agree the steps below:**

- the base-year switch (`YR_CAL_BASE` as a setting, FY2021 anchors, existing HIR projects as base-year management);
- the 2020-21 inputs and the tools that build them;
- the emission factor changes: energy and lifecycle fields off the agriculture row, livestock and crop factors on the NIR 2024
  basis with an exogenous agriculture series, HIR and Destocked regrowth at the gross Clean Energy Regulator rate, and HIR carbon
  revenue on issued units;
- an ABARES demand option;
- the remaining coarse-resolution fixes (section 4);
- an HIR water fix: the HIR water net-yield effect has its sign inverted and lacks the area factor, at every resolution.

All of it was written before your NetCDF change (`ce25e72e`), so each piece will be rebuilt on the latest `jinzhu` as its own PR.

## 2. Base-year design

We propose to agree this before either of us rebuilds inputs.

- **Base year FY2021 (2020-21)**, with financial-year labels (LUTO year Y is the financial year ending in Y). It is the most
  recent year with an ABS agricultural census, an NLUM epoch and a full NIR inventory, so the base-year map, production, head and
  emissions can all be checked against observation.
- **Map.** NLUM v7.1 2020-21 classes are allocated to LUTO cells within each SA2 (2021 boundaries), so each SA2 keeps its class
  areas: whole cells, a definition crosswalk from each NLUM class to a LUTO land use, and the cells where a class's share is
  highest go first. Eligibility follows the land uses that occur in each SA2. Irrigated crops are eligible only where NLUM
  records irrigation in 2010-11, 2015-16 or 2020-21.
- **Livestock.** Head by SA2 and type is the ABS 2020-21 count scaled to the NIR state totals. A per-SA2 stocking calibration
  places that head on the mapped pasture, and every NIR head is either on a cell or in a named row. Existing HIR projects (the
  Clean Energy Regulator register) start as base-year management.
- **Emission rows.** Two rows, as merged in #229: agriculture, and net LULUCF. History from NIR 2024 and projections from the
  DCCEEW 2025 baseline, AR5 GWP100, public sources only. The agriculture row binds from FY2025, and FY2022-24 serve as historical
  benchmarks.
- **Economics.** Farm-gate prices from ABS value of production. Costs from the ABARES Farm Data Portal for broadacre industries.
  For horticulture, cotton, rice, sugar and other non-cereal crops, the existing files' cost-to-revenue ratio is carried over.
  Constants set in 2010 dollars are rescaled by the ABS CPI.
- **Evidence, on our branch, not `jinzhu`.** Both results below come from our local integration of the switch and the 2020-21
  inputs, built on `jinzhu` before `47decf3e` and before your NetCDF change.
  - At RES1, the agriculture row reproduces NIR FY2021 sector 3 within 0.12 per cent: -90,646.7 t of 75,495,385 t nationally,
    with every difference attributed.
  - A run at RES10 from FY2022 to 2050 was optimal on the first attempt in every year and passed our acceptance checks. It used
    a carbon price of 0, no ag-management adoption above the base year, and GBF targets off. Recorded limitations:
    - six irrigation divisions sit outside 25 per cent of ABS water use;
    - wool reaches 0.787 of demand by 2050;
    - winter cereals reach 40.97 Mha under the default demand.

**Could you confirm the base year and map, or say what you would change, before anything is rebuilt?**

## 3. How the current block weighting works

Code: `Data.coarsen_guarded_arrays` in `luto/data.py`.

- **What it computes.** At RES > 1, each eligible coarse (lm, cell, lu) entry is replaced by a weighted mean of the fine values
  over a set of the block's cells:
  - the fine cells allocated to that (lm, lu) in the base year, where the block has any;
  - otherwise, the block's eligible land-use cells.

  Ineligible entries keep the representative cell's value.
- **The weights.** They depend on the field, so that each product nests and a block's coarse product equals its RES1 base-year
  total. There is no single block ratio.
  - Area for per-hectare fields: yield, costs, water requirement, crop GHG, pasture growth.
  - Area x yield for production fields: quantity, price, the crop climate multiplier.
  - Area x water requirement for the crop water price.
  - A chain for livestock: pasture growth, then safe utilisation, then capacity, then head. Per-head fields are weighted by head.
- **Your Apples example.** With no Apples allocated in the block, the cost is the area-weighted mean over the two eligible cells:
  (2 x 11 + 7 x 4) / (2 + 7) = 5.6 per ha. If Apples were allocated in the block, the mean would be over those cells instead.
- **Inputs it covers.** Agricultural economics; agricultural GHG for crops, livestock and irrigated pasture; pasture growth and
  safe utilisation; feed requirement or stocking calibration; the climate multiplier; crop and livestock water requirement and
  the water delivery price; and the stubble rate.
- **Read at the representative cell.** Natural-land and soil carbon, environmental and carbon plantings, water yield, savanna
  burning, biodiversity and the other non-agricultural inputs (#225, #226). Your plan to weight every input covers these.
  `REAL_AREA` is also outside the weighting: it is the representative cell's area x `RESMULT` (section 4).
- **Where it came from.** The averaging came in #202 and was extended in #230 (stocking calibration and stubble) and #231
  (allocated cells first). #227 is the NaN guard: it raises if a guarded array is NaN in an eligible entry, before any NaN-to-zero
  conversion, at every resolution.
- **#231 and the map.** #231 averages over the base-year allocation of whichever map is loaded. On `jinzhu` that is the 2010 map.
  It does not use the 2021 map, which is not proposed yet.

## 4. Coarse-resolution effects as tests for the refactor

Each was measured on `jinzhu` `47decf3e`, before `ce25e72e`, with 2010 inputs and default settings except `RESFACTOR` and
`SIM_YEARS`. Each fix leaves RES1 unchanged. Your coastal-block change alters the coarse cell set, so the figures will move. We
are happy to rerun all three on your branch at RES5 once it settles.

| Effect | Measured | RES1 check |
|---|---|---|
| **Block area where land-use area is meant.** `REAL_AREA` is the representative cell's area x `RESMULT`, so it counts the whole block, including land outside any land use | Land-use area overstated by 10.5, 16.2 and 23.7 per cent at RES5, 10 and 20. On the land-use basis, cap right-hand sides fall 9.5, 13.9 and 19.2 per cent | every `Data` and matrix array identical; no coarse land-use change at RES5, 10 or 20 |
| **Allocated area above the eligible fine-cell area.** A block is eligible if any fine cell is, and can then be allocated more than its eligible cells' area | 1,058 kha nationally by 2050 at RES5 (386 kha in 2030); 1,878 kha at RES10 (684 kha in 2030). Capping at the eligible area removes all of it | bound identical (0 entries changed) |
| **Destocked on mixed blocks.** Destocked GHG and biodiversity are gated on the block's single coarse land use, so mixed blocks lose removals | Full-adoption GHG -53.98 Mt at RES5 against -82.94 Mt with the selection fixed. RES1 is -82.94 Mt; RES10 -47.60 and RES20 -42.65 before the fix | every array identical |

## 5. Noticed in the merged code

- **GHG targets path.** `dataprep.py` runs `luto/tools/ghg_targets.py` with `--out` set to `input/ghg_targets`, while
  `Data` now reads `input/Nick_build/ghg_targets`. A fresh `dataprep` run writes the targets where the model does not look. This
  is a one-line PR if you would like it, or it can go with the move in section 6.

## 6. Proposals for your decision

- **Data-building code out of `luto`.** Move `ghg_targets.py` and `ag_mgt_caps.py` out of `luto/tools/`, with their calls in
  `dataprep.py`.
- **A separate repository for our input tools.** They would write NetCDF into `input/Nick_build/` in your layout, so LUTO only
  gains the settings to read them.
  - Inputs are rebuilt from public sources where one exists (ABS, ABARES, NIR, DCCEEW, the Clean Energy Regulator). The rest
    are carried from the existing input build and recorded as such.
- **Order of the remaining PRs**, each one change from the latest `jinzhu`:
  1. the GHG targets path (section 5);
  2. the HIR water fix (independent of coarsening and the base year);
  3. energy and lifecycle fields off the agriculture row;
  4. once the inputs are in `input/Nick_build/`: the NIR-basis livestock and crop factors, HIR and Destocked regrowth at the
     gross rate, then HIR carbon revenue, each needing its input file;
  5. the base-year switch, after we agree section 2;
  6. the ABARES demand option.

  The coarse fixes in section 4 wait for your refactor and serve as its tests rather than as PRs.

## 7. Load time

Issue #213 measured the same overhead before your refactor. At RES10, a `Data` build went from 79 s to 346 s with the block
weighting, of which `coarsen_guarded_arrays` was 270 s, and peak memory went from 5.98 GB to 6.14 GB.

It proposes caching the weighted arrays per `RESFACTOR`, as `match_GBF8_bio_layers` already does for GBF8:
- write the cache to a temporary file and rename it into place;
- key the cache on the settings that change the arrays and the hash of each input file read.

This would sit alongside parallel reads: the first build at a resolution pays the cost, and later builds read the cache.
