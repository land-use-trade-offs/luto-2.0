# LUTO2 Futures Log

A running record of to-do items and improvement plans. `FINDINGS.md` records what has been found; this file records what we intend to do.
Entries are in **descending date order** (newest first). Each entry stands on its own.

---

## 20261110 — Vue report: a Hypothesis tab (inputs and assumptions) and a terminology wiki

### TL;DR

Two new features for the dashboard (Jinzhu, 2026-11-10). Noted only: the design of each comes from a walkthrough with Jinzhu, not yet held.

1. **A Hypothesis tab.** A new page that plots the model's input data and assumptions: climate change, demand, yield, water, economics, and the other inputs. The reader then sees what went into a run, not only what came out.
2. **A terminology wiki.** Every term the report uses gets an explanation, with a link to Wikipedia or to an illustration of the concept.

### Why

The report shows results with no view of the inputs or assumptions behind them, and it uses terms a returning reader or someone from another field cannot be expected to know.

### Open

- Everything below the two headlines: which inputs, at what level (national / region / cell), which chart types, where the terms are kept, how a term is opened from a page.

---

## 20261109 — GBF8 suitability with range masks: 5 km suitability + fixed weights `W` + sparse range levels

### TL;DR

The species suitability that GBF8 reads is about to grow new axes: a range mask per species (EOO or AOO), a dispersal option (FLAT or SPECIES) and, for 221 plants, a state mask. Every combination is the same suitability times 0/1 masks, so no combination should be stored.

**Preferred plan:** store only what is truly at each resolution.
- **Suitability stays at 5 km**, as now: 4 GB per SSP for 10,597 species × 5 years.
- **One fixed bilinear weight matrix `W`** (cells × 5 km pixels, 4 weights per cell, ~0.3 GB) turns a 5 km layer into the cells: `W @ layer`.
- **The range is stored sparse on the 1 km cells,** once per species × mask type: one small integer per cell, the level of the smallest buffer that reaches it. It is the same for every SSP and year. Measured for the 1,311 vertebrates, both masks: **5.3 GB** compressed.

At run time a combination is `(W @ suitability) × (level ≤ the level of rate × (year − 2010))`, computed for a block of species when its rows are due. This replaces `xr.interp` and the per-(SSP, RESFACTOR) cache files with one sparse multiply. It is ~25× smaller on disk than storing every combination, and ~8× smaller than storing the suitability sparse at 1 km.

### Why

**Now:** `bio_GBF8_ssp<SSP>_EnviroSuit.nc` holds 10,597 species × 5 years on the 5 km raster (808 × 978, uint8): 4 GB. `match_GBF8_bio_layers` interpolates it linearly to the cell centres (`xr.interp`) and caches the result per SSP and RESFACTOR: the RES5 cache is 10.3 GB, RES10 3.1 GB.

**A native 1 km suitability file is not the answer.** Measured 2026-11-09 on 20 species spread over the list (one year, linear interpolation to all 6,956,407 cells):

| storage | per layer | all 10,597 species × 5 years |
|---|---|---|
| now: 5 km 2-D raster, uint8, zlib 9 | 0.077 MB | 4 GB |
| 1 km cells, uint8, zlib 9 | 6.2 MB | ~330 GB |
| 1 km cells, float32, zlib 4 | 11.0 MB | ~590 GB (the size of a RES1 cache) |

It is ~80× bigger, not 25×: the 5 km raster's zero and sea areas compress to almost nothing, and interpolation gives every cell its own value, so 1 km barely compresses. And it holds no new information: the MaxEnt suitability exists only at 5 km.

**Sparse storage of the suitability saves little.** Measured 2026-11-09:
- Unmasked, a layer is non-zero on ~30 % of land on average (200 species, 1990 and 2090; median 14–18 %, 90th percentile 73–76 %). A sparse entry costs 5–8 bytes (value + int32 cell index) against 1 (uint8) or 4 (float32) dense: ~1.7× smaller than float32, *larger* than uint8.
- Masked, over each species' envelope (the cells within its largest buffer with suitability > 0 in some layer), the vertebrates hold 1.83 G entries for EOO and 1.81 G for AOO. Birds are 70 % of them: their envelope averages 41 % of land (amphibians 4 %, reptiles 7 %, mammals 19 %). Stopping the buffers at 2050 cuts only 14–18 %; the suitability > 0 filter only ~9 %.

**Disk per SSP, 1,311 vertebrates, both masks:**

| design | size |
|---|---|
| every combination stored dense, 1 km uint8 (2 masks × (historic + 2 dispersal × 4 years) = 18 layers per species) | ~164 GB |
| one dense unmasked 1 km layer per species × year | ~46 GB |
| envelope: suitability sparse at 1 km (cell index + distance + 5 years; uncompressed, CSR / COO) | 40 / 47 GB (EOO alone 20 / 24) |
| **preferred: 5 km suitability + `W` + sparse range levels** | ~0.5 GB suitability (the vertebrates' share) + 5.3 GB range levels for both masks, shared by all SSPs and years (compressed) + 0.3 GB `W` |

The envelope copies each 5 km suitability value into ~25 cells and again for every year and SSP; the preferred plan stores the 1 km part — the range — once.

**In memory at RES1 the base is ~100 GB or more.** All species on all cells is 6,956,407 × 10,597 ≈ 74 billion values per year: ~74 GB as uint8, ~295 GB as float32, and the year blend reads two years. Neither design changes that; species must be computed in blocks.

### Design

```
value(cell, species, ssp, year, mask, dispersal)
    = (W @ suitability_5km[species, ssp, year])[cell]
    × 1[ level(cell, species, mask) ≤ level of the buffer rate(species, dispersal) × (year − 2010) ]
    × 1[ cell in the species' native states ]          (only the plants that carry a state mask)
```

| part | depends on | storage |
|---|---|---|
| suitability, 5 km raster, uint8 | species × SSP × year | the current file, 4 GB per SSP |
| `W`, bilinear weights, sparse (cells × 5 km land pixels) | the two grids only | ~28 M entries, ~0.3 GB, once |
| range levels, sparse on the 1 km cells, uint8 | species × mask (EOO / AOO) | 5.3 GB for the vertebrates, both masks (compressed); plants add less per species (no buffer) |
| level → buffer km table | species | small (≤ 7 levels per species) |
| dispersal rates | species × dispersal | a table (Carla's dispersal sheet) |
| state mask (native states + 20 km) | 221 plants | small; can fold into the range levels |

**The range levels.** For each species and mask, each cell within the species' largest buffer gets the level of the smallest buffer that reaches it: 1 = the 2010 range, 2 = the next buffer, …; cells beyond every buffer are not stored. A (dispersal, year) mask is `0 < level ≤ k`, with `k` the number of the species' buffer distances up to `rate × (year − 2010)`. One array serves every year, SSP and dispersal option. The Zonation pipeline's step 3 already builds exactly this, per taxon, as scipy CSR (species × cell, uint8).

**CSR or COO.** Either. As COO in xarray (dim `entry`; `species` int16, `cell` int32, `level` uint8, sorted by species) a species' entries are one slice found by `searchsorted`; the species column costs 2 bytes per entry in memory (7 bytes per entry against CSR's 5: cell int32 + level uint8) and almost nothing on disk, since zlib compresses its long runs. CSR in xarray is the same plus a `start` variable on a `species` dim (the `indptr`).

### Plan

1. **`W` in dataprep.** Build the sparse bilinear matrix from the 5 km raster's transform and the cell centres (`COORD_LON_LAT_2D_FULLRES`). Linear interpolation on a regular grid is bilinear, so `W @ layer` reproduces `xr.interp(..., method='linear')`; match its edge handling once, then check against the RES5 cache cell by cell.
2. **Replace the cache.** `get_GBF8_bio_layers_by_yr` reads a block of species' 5 km layers for the two years around `yr`, multiplies by `W[MASK]` (the rows of the run's cells; RESFACTOR > 1 takes its representative cells), blends the years as now. `match_GBF8_bio_layers` and the `_cache` files go.
3. **Range levels as an input.** One file per mask type (EOO / AOO) with the level → km table; settings choose the mask (none / EOO / AOO) and dispersal (FLAT / SPECIES). Apply the mask after `W`, on the block.
4. **Stream species in blocks.** Row generation already builds GBF8 rows lazily in batches; a block is computed when its targets are due, never all species at once.
5. **Species list.** Move GBF8 to Carla's final native list (8,833: 1,314 vertebrates + 7,519 plants) with its targets; the 221 plant state masks follow it.

### Open points

- **Edges.** `xr.interp` gives NaN where a neighbour pixel is NaN; `W` must do the same, or renormalise over the valid neighbours. Decide and document which.
- **Nearest instead of bilinear** is a single index lookup (cell → pixel), cheaper still, but changes the values slightly from today's linear interpolation.
- **Plants not measured.** Their EOOs are not built yet. With no buffer, a plant's range levels are just its base range; if ranges are like reptiles' (~4.5 % of land each), 7,519 plants add ~2.4 G entries per mask.
- **AOO envelopes** are the record cells plus the buffer: tiny for plants, large for wide-ranging birds. Outlier records stretch both EOO and AOO.
- **Considered and set aside: the envelope design** (suitability interpolated in dataprep and stored sparse at 1 km over each species' envelope, so no interpolation at run time). Measured above at 40–47 GB per SSP for the vertebrates' two masks, ~8× the preferred plan; worth it only if the run-time `W @ layer` proves too slow, which a sparse multiply should not be.
- **Zonation is different.** Its inputs must be finished rasters (z5 reads files; a VRT cannot apply the mask formula), so a Zonation run table does store one layer per species × run. Its outputs are one rank map per run (~750 runs for 9 taxon sets: 1–3 GB as uint16 on the cells) and need no interpolation in LUTO.

---

## 20261102 — show uncertainty: an ensemble of perturbed solves, once grouping makes solves cheap

### TL;DR

An LP gives one sharp answer, and options that are almost equally good can swing on a small nudge. Cell grouping could make many solves affordable, so a run could report a range for each result instead of one number.

### Why

The first grouped test (RES1, 2045 → 2050, with GHG off) came within 0.0154 % of the full objective, and most land uses moved under 0.2 %. Yet dry Winter cereals under Biochar fell 11.2 % (−50,731 ha). The cause is not established. A near-tie is one explanation, and that is what this work would test.

### Plan

1. **Small pilot first.** One year, about 10 solves, with the objective and yields perturbed by about 1 %. See which results move: land use, ag-management, or both.
2. **Perturb what is actually uncertain, in a correlated way:** commodity prices, yields by region, costs, climate and demand scenarios. Independent noise per cell averages out over a region and shows little.
3. **Group once per scenario, on the unperturbed model.** Perturbing the grouped tables directly (for example, scaling objective coefficients by column type) avoids rebuilding the tables each time. Check that the blobs stay valid under the perturbation.
4. **Report ranges** (for example 5–95 %) for each land use by region, and flag results that swing.

### Open points

- **Separate grouping error from input uncertainty.** Run the same perturbations on a few full solves, or compare against a tighter tolerance. Otherwise grouping's own error reads as uncertainty.
- **Cost.** A trajectory is about 7 steps, and each still needs the full tables built once. At the first test's speed, about 30 min per solve on Gadi-equivalent CPUs, a 50-member ensemble is about 175 h of solve time before table builds, so members must run in parallel.
- **Multi-year paths.** A perturbation in one year changes the next year's base, and so its blobs. Perturb one fixed set of factors along each path.
- **Alternative for near-ties:** near-optimal search, which maximises a land use while holding the objective within a small margin of the optimum. It shows how much can swing at almost no cost, without assuming any input distribution.
- **Depends on** grouping holding up on harder scenarios, such as binding GHG, and over several years.

---

## 20261102 — review Nick's merged PRs: walk every commit, leave no unknown function

### TL;DR

Nick van Schoten's PRs were merged into `jinzhu` on 2026-10-19 and 2026-10-22, but too many conflicts meant they were merged **without a code review**. Only #202's code was read closely; the others were judged from their PR descriptions, reference checks and test builds. That is **36 commits and about 3,400 added lines** (validation scripts included), now running in every study.

The goal is to walk through all of them and understand them, so that no function, setting or check in the model is unknown to its owner.

**Finding the right commits.** There are two Nicks, and `--author=Nick` mostly finds Nick Williams (Biarri, 2024–25), who is out of scope. Use:

```bash
git log --no-merges --author='van Schoten\|nickavs\|nicholas.vanschoten'
```

This gives 44 commits: the 36 merged in October, plus 8 earlier ones as `nickavs` (2025-11-27 to 2026-02-10, the first renewable-energy implementation). Their review state is unknown, so walk them last. Five of them also add `raise` / `assert`.

**PR numbers changed.** #203–#206 and #222 were reopened as #227–#231. Start from the merge commits, not the old numbers: `3ee56aa6`, `9552f57e`, `205c4442`, `0e57fb08`, `56a62ddf`, `8af70036`, `cf9886b1`, `b1d21434`, `fdbfcef2`, `a9568fae`, `948d88b0`.

**Where to look first.** Files changed in the most of these commits: `data.py` (25), `settings.py` (12), `economics/agricultural/quantity.py` (10), `economics/agricultural/ghg.py` (10), `economics/agricultural/revenue.py` (8), `economics/agricultural/cost.py` (7), `solvers/row_inputs.py` (6).

### What prompted this

On 2026-11-01 two RES1 write jobs failed after about 4 h, costing about 390 SU each, with `Transition GHG 2020: output total != GHG row term` (T150: 36840.9 vs 36831.1). This check had been merged unreviewed. Nick added it inside a fix commit, `e377dd16` (#217, 2026-10-09, "charge land clearing its full stock loss"): `write_ghg` compares its total with the `GHG_TRANSITION` value stored by `simulation.store_solution`.

- **Cause: float32 drift, not `get_base_held`.** `simulation.py:351` sums with `np.dot` over float32 arrays, while the writer sums in float64. A check job reproduced the stored value exactly in float32, and recomputing it in float64 matched the writer's total to 1e-8 in every year, which leaves no gap for anything else to explain. The drift reached 2.7e-4 relative, against a 1e-5 tolerance. RES5 passes, with 25× fewer entries.
- **`get_base_held` was suspected at first, and the evidence clears it.** It was already in the code before the check was added (moved to `tools` in `5581ff31`, 2026-09-16). It and `SOLUTION_SHARE_FLOOR` do deliberately change small values, so a tight check elsewhere could still trip on them, and the review should test for that.
- **What it left behind.** `6d4891ca` widened the tolerance to 1e-3, and `store_solution` stays in float32 by decision. The check can no longer see a real mismatch under 0.1 %, and it still runs only hours into a write.
- **Doc mismatch.** `CLAUDE.md` says `store_solution` computes GHG "in float64", but the code uses float32 `np.dot`. Correct that line.

**The lesson for the review:** a check can pass on every run its author tried and still fail in a later study. The dangerous ones compare two totals tightly, stop a run late, or sit inside a commit about something else. This one was hidden because the drift exceeds the tolerance only at RES1.

### Rules for the walk

- **One commit at a time, in the order below,** reading the commit's own diff with `git show <hash>`. Read the current code as well, because the merge changed it.
- For each commit, answer and record:
  1. What does it change in the model's numbers? Is it on by default?
  2. What new functions, settings and input files does it add? Who calls each one? Is it still needed?
  3. What can it stop? For every `raise` / `assert`: when does it fire, how long into a run, and is its tolerance safe against float32, `get_base_held`, `SOLUTION_SHARE_FLOOR` and `SOLVER_COEFF_MIN`?
  4. For anything comparing two totals: was it only ever run at coarse resolution?
  5. Do I agree with it? Keep, change, or remove.
- **Review the merges too.** Conflict resolutions are changes as well, so read `git show --cc <merge>` alongside Nick's commits. `948d88b0` (Nick's stack + S2 / S6) resolved 27 conflict hunks across 9 files, and that merge's diff touches 29 files. Each resolution is new code that neither side wrote or reviewed as it stands.
- Fix wrong numbers in separate commits, and rerun the same small case before and after.
- **Local only:** no pushes, comments or merges on GitHub until Jinzhu says so.
- **Dead code:** report it with evidence, and delete only on Jinzhu's decision.
- **Validation scripts:** flag any of Nick's test or check scripts found in `luto/` for removal. `check_nesting.py` and `check_nan_guard.py` are already out (`cbb946f2`).
- **Gadi:** no data work on the login node. Copy data back and test locally.

### Checks that can stop a run

Found by searching the 36 October commits for added `raise` / `assert` in `luto/`. Checks that only print are not in it, and the 8 earlier commits are still to add. To regenerate the list:

```bash
git log --author='van Schoten\|nickavs' --no-merges -p --format='COMMIT %h' -- luto \
  | grep -E '^COMMIT|^\+\s*(raise |assert )'
```

| When it fires | Where | Added by | Review question |
|---|---|---|---|
| **During write, hours after the solve** | `write_ghg`, transition GHG total vs stored row term | `e377dd16` | Should it warn instead, run at store time, or go? It costs the most when it fires |
| Data build | `Data.check_eligible_nan` | `5797d518`, `408933d3`, `1724c4b6` | Conflicts with the rule that a NaN input means "not allowed" (zero upper bound), not an error. Is the coarse build safe without `coarsen_guarded_arrays`? |
| Each solve year's row inputs | `Data.check_ag_man_nan` | `725f2a16` | Same; it also fires later than the data build |
| Data build | `load_ghg_targets`, target series coverage and shape | `cdc90a0d` | Do the messages still name the right directory? The series now sit in `input/Nick_build/ghg_targets/` |
| Data build / solve year | Observed adoption caps | `97b42747` | Fires when a cap starts after the year, relevant to `SIM_YEARS` choices |
| Settings | `GHG_BENCHMARK_ROWS` | `669ad140` | Low risk |
| Data build | Water yield file years | `1b14e523` | Low risk; confirm with a changed base year |
| Data build | `coarsen_guarded_arrays` weight class; land-cover type | `1f858af0` | A new input field stops the coarse build |
| `dataprep.py` only | `ghg_targets.py`, `ag_mgt_caps.py`: sha256 pins, sum checks | `05878103`, `67de3ed2`, `c671c96d` | Do not affect a run; they affect rebuilding the inputs |

### The commits, in reading order

Start with the small independent fixes, then the two that change the model most.

| PR | Commit | What | New functions to understand |
|---|---|---|---|
| #201 | `a43f0ee2` | Coarse eligibility counts land-use cells only | |
| #219 | `201928c5` | Credit Savanna Burning abatement at the carbon price | |
| #215 | `76fc0804` | Asparagopsis credit goes to the land management it was computed for | |
| #228 | `0d699a26` | Coarse stream length is the block total, density over the block | |
| #217 | `e377dd16` | **Charge land clearing its full stock loss, spread over the step; adds the write check** | |
| #217 | `3bf7506b` | Comment rewording | |
| #202 | `1f858af0` | **Coarse cells aggregate economics, GHG, pasture and climate inputs** | `coarsen_guarded_arrays`, `block_mean`, `weight_class`, livestock accessors |
| #202 | `5dc1db27` | `check_nesting` validation script | Moved out of `luto/` by `cbb946f2` |
| #231 | `74a8a68c` | Coarse cells average allocated entries over their base-year cells | `block_mean`, `cell_set` |
| #231 | `76a27956`, `8d5c7248`, `43a56c94`, `e25850bd` | `check_nesting` extensions | Moved out of `luto/` |
| #231 | `01a66c81` | HIR baseline counts project area on the option's own cells | |
| #231 | `1724c4b6` | NaN guard checks `FEED_REQ` only without a per-type k file | |
| #227 | `5797d518` | NaN guard on the arrays the ag economics read | `check_eligible_nan` |
| #227 | `725f2a16` | NaN guard on the ag-management effect matrices | `check_ag_man_nan`, `raise_nan_report` |
| #227 | `493a7912` | `check_nan_guard` injection tests | Moved out of `luto/` |
| #229 | `05878103` | **Build the split GHG target series from public sources (612 lines)** | `tools/ghg_targets.py` |
| #229 | `1b14e523` | Read every base-year-dependent input by year label, not by offset (8 files) | `get_water_yield_file_row` |
| #229 | `cdc90a0d` | **Split the GHG constraint into agriculture and net LULUCF rows (9 files, +476)** | `load_ghg_targets`, `get_ghg_matrices_parts`, `get_irrpast_energy_ghg_matrices`, `sum_parts`, `side_of_parts`, ag-mgt effect parts |
| #229 | `67de3ed2` | Unmodelled agriculture series for the agriculture row | |
| #229 | `623a479b` | Move direct clearing from modelled to exogenous LULUCF | |
| #229 | `91408d9b` | Plantings baseline rate from NIR FY2016-21 | |
| #229 | `c671c96d` | Build observed ag-mgt adoption caps and the existing HIR project register | `tools/ag_mgt_caps.py` |
| #229 | `97b42747` | Cap ag-management adoption at observed adoption | `AG_MANAGEMENT_OBSERVED_CAPS` |
| #229 | `7dbbd338` | Count LUTO's HIR on both GHG rows above the existing HIR projects | `load_hir_baseline_area`, `get_hir_baseline_t`, `HIR_BASELINE` |
| #229 | `669ad140` | Hold either GHG row as a benchmark in projection years | `GHG_BENCHMARK_ROWS` |
| #229 | `edddf600` | Irrigated pasture energy sub-term reads `agghg_irrpast` per (…) | |
| #230 | `3f05eb5f` | **Per-type stocking calibration k in place of `FEED_REQ`** | `LVSTK_K_FILE` |
| #230 | `f03a0b67` | **Sheep grazing cereal stubble as a co-product of the cereal land use (8 files)** | `get_stubble_yield_pot`, `get_quantity_stubble`, `get_cost_stubble`, `get_ghg_stubble`, `get_stubble_ghg_matrices`, `STUBBLE_DSE_FILE`, `STUBBLE_LAND_USES` |
| #230 | `81b20963` | One set of stubble sheep products per stubble land use | |
| #230 | `4cfea2f8` | Coarse aggregation of k and the stubble rate | `lvstk_k`, `stubble` |
| #230 | `408933d3` | NaN guard covers k and the stubble rate | |
| #230 | `24984b54` | Validation scripts cover k and stubble | Moved out of `luto/` |
| #230 | `8e0f1e9e` | No sheep per-head data means no stubble sheep | `get_stubble_effective_rate` |
| merge | `948d88b0` | **Conflict resolutions: Nick's stack + S2 / S6** | |

### Open questions carried over

- Is `coarsen_guarded_arrays` needed at all? The options are sampling plus marking NaN entries ineligible, simple block averaging, or chained weights. It costs build time: the RES10 `Data` build went from 79 s to 346 s by Nick's figures. #227 passes at coarse resolution only with this aggregation, and #230 depends on it too.
- Replace the livestock accessors with plain data attributes (proposed, not done).
- Refactor `coarsen_guarded_arrays` into plain sections (only partly done).
- Audit every model-path NaN fill (`fillna` / `nan_to_num`): where the NaN comes from, what the fill means, and a verdict. Not started.
- `GHG_EMISSIONS_LIMITS` is `'low'` in `settings.py`, so the two-row GHG model is the default. Confirm this is intended for the studies.
- **Results are not comparable across the merges.** Coarse-resolution runs before and after `47decf3e` differ, and the planned RES1 A/B check on Gadi was not done before the 2026-10-22 push. Decide whether it is still worth running.

### Done when

- Every row above is marked keep / changed / removed, with a one-line note on what it does.
- Every check that can stop a run has a decision on whether it stays, and its tolerance is justified against the small-value changes above.
- The GHG target series and stubble logic can be explained without opening the PRs.
- Findings that change results are recorded in `FINDINGS.md`.

---

## 20261102 — report dashboard: readability and interactivity

### TL;DR

The report's numbers and wording have had a round of fixes, but it still does not explain itself. Three things are wanted:

1. A help button that explains the model's terminology.
2. An assumptions page that explains what LUTO assumes.
3. The year slider, the main-page Australia map and the selection buttons brought together, so the user sees how the choices nest.

### Where it stands

- **Done:** review fixes to card numbers, units, the GHG sign, target lines and map loading; GBF3 / GBF4 % over each item's own pre-1750 total; per-year target lines (`7712bf9a`, `65d7e9ee`).
- **Tried and reverted:** a broad visual redesign driven by Claude's design skills. The result was not satisfactory, so none of it was kept.
  - **What this means for the next attempt:** make one change at a time, starting with a mock-up of one page (GHG suits) that is agreed before any template changes. Do not restyle all views in one pass.
- **Code today:**
  - Views are in `luto/tools/report/VUE_modules/views/`, and every page repeats its own selector markup.
  - No view or component mentions a glossary, terminology or assumptions.
  - The year has its own card, apart from the Category / Ag Mgt / Irrigation / Land use card (`GHG.js:291`, similar on the other map pages).
  - The year drives only the map. The chart always shows every year, does not mark the selected one, and has no click events (`components/chart_container.js`).

### 1. Terminology help button

- One always-visible help button in the app shell, so every page gets it without per-view code. It opens a searchable side panel that does not cover the selectors.
- One data file, for example `resources/glossary.js`, holds each entry: term, plain-language definition, unit, and the page(s) it appears on. The panel and any later inline tooltips read this one file.
- Start with one vocabulary used everywhere: agricultural commodities, ag-management solutions, non-agricultural land uses, net economic returns, net emissions, net water yield. Also cover the terms a returning reader trips over: GBF2 / 3 / 4 / 8, SNES, ECNES, NVIS, NRM, HIR, dvar / "proportion of cell", simulated versus interpolated years.
- A modeller reviews the definitions. They are claims about the model, not UI text.

### 2. Assumptions page

- A new route and sidebar entry, separate from Settings. Settings lists raw parameters; this page says what they mean for the results.
- One section per theme. Each states the assumption in plain language, then the value **this run** used, read from `Supporting_info.model_run_settings` (already loaded by `Settings.js`), so the page cannot disagree with the run:

| Theme | Settings to explain |
|---|---|
| Climate change | `SSP` (scenario, RCP), `CO2_FERT`, `WATER_CLIMATE_CHANGE_IMPACT` |
| Demand | `DIET_DOM`, `DIET_GLOB`, `CONVERGENCE`, `IMPORT_TREND`, `WASTE`, `FEED_EFFICIENCY`, `DYNAMIC_PRICE` |
| Water | `WATER_LIMITS`, and how net yield inside and outside the study area is counted |
| GHG and carbon | `GHG_EMISSIONS_LIMITS`, `CARBON_EFFECTS_WINDOW`, `RISK_OF_REVERSAL` |
| Biodiversity | `GBF2_TARGET`, `GBF3_NVIS_TARGET`, `GBF4_TARGET_SNES`, `GBF4_TARGET_ECNES`, `GBF8_TARGET` |
| Renewables | `RENEWABLES_OPTIONS`, the target and spatial-layer scenarios |
| Economics | `DISCOUNT_RATE`, `AMORTISE_UPFRONT_COSTS`, `AMORTISATION_PERIOD` |

- The prose is written and checked by the modellers. Claude can draft from the code, but a draft is not a source for what the model assumes.

### 3. Link the year, the map and the selections

The controls should read as one sentence, from the broadest choice to the finest, and the chart and map should visibly respond to the same state:

```
Region     [State | NRM]  [▾ Victoria]
Year       ◀ 2030 ▶   ○───○───●
──────────────────────────────────
Category   [Sum | Ag | Ag Mgt | Non-Ag]
Ag Mgt     [▾ ...]
Irrigation [All | Dryland | Irrigated]
Land use   [▾ ALL]
```

- **One query panel:** merge the year card and the category card. Where and when come first because they apply to everything below; what comes after the divider. A row that does not apply disappears without the panel jumping.
- **Year joins chart and map:** the selected year is highlighted on the chart (a Highcharts `xAxis.plotBands` band), and clicking a column sets the year (`point.events.click`). This is the cheapest fix and needs no data change.
- **One summary line above the map:** `GHG › Ag Mgt › All irrigation › All land uses — Victoria, 2030`. It also becomes the chart and export title.
- **Home page:** region, year and theme sit in one bar above the content, and the map's job is to pick the region.
- **Later:** year stepper and play button, selection kept in the URL, region click on every map page.

### Order of work

1. Year ↔ chart link. It is small and can ship alone.
2. Mock-up of the query panel on the GHG page, agreed first, then the other views.
3. Help button and glossary.
4. Assumptions page, once the modellers have supplied the text.

---

## 20261102 — cell clustering: merge similar cells into one giant column before the solve

*Design discussion of 2026-10-27, against `jinzhu` at commit `8ae24e5`. Nothing is implemented, and every performance figure below is an estimate, not a measurement.*

### TL;DR

Merge RES1 cells with similar coefficients into clusters after the row and column tables are built. The aim is to **solve at RES3–RES5 size (about 10–25× fewer cells) with all of the RES1 information**, and eventually retire `RESFACTOR` sampling and its corrections.

| Compression | Equivalent solve size |
|---|---|
| ~9–10× | RES3 |
| 25× | RES5 (the usual debug scale) |

### Core idea: aggregation as an exact matrix transform

Clustering happens after the tables exist: `get_rows` → `row_bounds` → **cluster** → solver.

```
A_agg = R · A · P      obj_agg = Pᵀ · obj      rhs_agg = R · rhs      x = P · x_agg
```

- **P** (cols × new cols, 0/1): merges the columns of one cluster that share a column *type*, meaning every column field except `cell` (`block, m, j, k, am_idx, j_idx, from_m, from_j, from_k`).
- **R** (new rows × rows, 0/1): sums each cluster's per-cell structural rows (node balance, source cap, ag-mgt link, renewable ceiling). Wide rows (demand, GHG, water, renewable, regional adoption, GBF) stay as they are.
- **Bounds:** share lb / ub are unchanged; `base` is summed through the node-balance rows.
- **Why it is exact:** the inputs are already per-cell totals (cost etc. × `REAL_AREA`) and the variables are shares. With equal shares in a cluster, Σ aᵣ·x = (Σ aᵣ)·x, so summing columns is exact and no area-weighted averaging is needed.

**Guarantee.** Aggregation is the same as adding the constraint "all cells in a cluster take equal shares":

- Any aggregated solution, mapped back, is **feasible** for the full problem.
- Under maxprofit, the aggregated objective is a **lower bound** on the full optimum.
- The only error is lost optimality from differences within a cluster. It never affects feasibility.

This is stronger than the 2023 Biarri `Clustering` branch, which hashed per-cell inputs and used one representative cell per cluster. That is wrong when cell areas differ.

**Rejected for now:** defining cluster variables as *area* rather than share. That gives a relaxation instead of a restriction: no forced equal shares, but no feasibility guarantee once mapped back to cells.

### What must be identical within a cluster

**A. Structurally required.** Without these, aggregation is invalid.

1. **Same base-year state**, in practice the same `(from_m, from_j)` or `from_k` source. Otherwise equal shares and node balance contradict each other and the arcs cannot be aligned. LUTO already provides the split: `transitions.get_base_dvar_mj_cell_map` → `{(m, j): cells}`.
   - Use only **pure** cells (cells under exactly one source). Mixed or fractional cells each form their own cluster. At 2010 RES1 nearly every cell is pure.
   - Non-ag sources are grouped separately by `k`.
2. **Same column set:** same `EXCLUDE` / no-go pattern (`get_ag_allowed_mrj`, which varies by SA2), GBF2 / MNES renewable exclusions, savanna eligibility, existing ag-mgt state (especially irreversible renewables).

**B. Quality only.** These never break correctness; mixing them costs optimality.

- **Region labels** (state, NRM, water region, IBRA, regional-adoption cell sets). Mixing them is still exact, because R·A·P splits each cluster's contribution across the regional rows correctly. They are cheap to add to the key, so including them is a sensible default.
- **Species** (see below).

**Transitions need no special handling.** Within one source stratum all cells share the same arc types, so arc columns (`flow_obj_*`, `trans_ghg_ag2ag`) are summed like any other column, and P already carries `from_*` in the column type. The per-source arrays are indexed by `local_r` in `src_map[src]` order, so they can also be aggregated in place with `np.add.reduceat` if aggregation is later pushed down into the inputs.

### Clustering features and method

**Features.** Take each cell's signature from the coefficients in A that the solver actually sees, divided by `real_area`. Rows are already rescaled around `RESCALE_FACTOR`, which largely removes the unit and scale problem.

**Options considered:**

- **Cosine similarity: rejected.** It ignores magnitude, and magnitude matters to the objective.
- **Global k-means on standardised features.** Works, but is slow at large k.
- **Per-input quantisation + exact intersection (preferred start).** Bin each input separately (0 as its own bin), concatenate the bin labels with the strata key, then `np.unique`. There is no distance computation and no scale problem, and the error is controlled per input. The risk is combinatorial growth with many inputs; use fewer bins for unimportant inputs and merge tiny groups into their nearest neighbour inside the stratum.
- **Per-coefficient ±1 % tolerance: too strict.** High-dimensional signatures rarely match on every dimension (estimated 1.5–3× compression only). Use absolute tolerances on scaled coefficients, PCA to the true dimension, and weight dimensions by their effect on the objective.

**Speed (estimates).** Everything except naive k-means takes seconds to minutes at RES1. Avoid sklearn k-means with k in the hundreds of thousands.

| Step | Approach | Estimated time |
|---|---|---|
| Signature from A | sparse slicing | tens of seconds |
| PCA | randomised, fit on ~1M sample | 1–2 min |
| Strata | int64 key + `np.unique` | seconds |
| Clustering | quantise + `np.unique` / recursive bisection / faiss-GPU k-means | seconds–minutes |
| P, R | `np.ravel_multi_index` → `pd.factorize` (avoid `np.unique(axis=0)` on huge arrays) | ~10 s |
| R·A·P | two sparse products, ~86M nnz | tens of seconds |

### The averaging problem and how to fix it

**Problem.** If a cluster of 10 cells should convert one cell's worth, the aggregated solution converts 0.1 of every cell. Then:

- each cell gains an extra source next year, so columns and arcs grow;
- fractional base states prevent re-clustering, so clusters lock up;
- irreversible options get spread thinly across many cells.

**Why it is limited.** An LP vertex solution has at most as many fractional variables as binding wide rows, so only a few thousand clusters are fractional and the rest are whole 0/1, which maps back to 0/1 cells. **Crossover or simplex is required:** a barrier solution without crossover is interior, and nearly every cluster would be fractional.

**Step 1: numpy greedy de-averaging (default).** For each fractional cluster, keep the converted area fixed (share × cluster area) and assign it to whole cells in descending order of reduced cost `d = obj − Aᵀπ`, using the aggregated LP's duals. This needs only sort + cumsum. Each cluster ends with at most one fractional cell, and area is conserved exactly.

- Side effect: wide rows can shift slightly, because coefficients differ within a cluster. Check with `viol = A_wide @ x_new − rhs`, and accept within tolerance (for example 1e-4 relative).

**Step 2: polishing LP (fallback, only if the violation exceeds tolerance).** This is **one** LP, not thousands.

- Free all columns of every cell in a fractional cluster (whole cells, so their structural rows are self-contained). Fix all other cells at the aggregated solution.
- Slice it straight from the existing RES1 tables, with no rebuild through `row_inputs` or the builders:
  ```python
  rhs_p = rhs - A[:, ~free] @ x0[~free]      # fixed contribution moved to the rhs
  LutoSolver(cols_p, rows_p, A_p, obj_p, inputs)   # same solver, unchanged
  ```
- **Objective:** the original objective on the free columns (the fixed part is a constant).
- **GHG and every other wide row:** the free cells get the remaining budget. Example: global cap 100 Mt, fixed cells emit 99.5 Mt, so the free cells' row is ≤ 0.5 Mt. All fractional clusters share this one row, so budget can move between them. Do **not** set a separate target per cluster.
- It is always feasible (the averaged solution satisfies it) and never worse. It needs a vertex solution.

After de-averaging, nearly all cells are 0/1 again, so the next year can **re-stratify and re-cluster from scratch**. Clusters may also only be split (never merged across strata) in later years.

### Species (GBF4 / GBF8)

GBF8 has ~10k species, ~3k after prescreen. Suitability is 0–100, and most species' pixel values are similar.

- Species never affect correctness. A GBF row is a linear sum over cells, so splitting a species range across land-use strata is fine (finer clusters can only improve the objective).
- The real danger is the opposite: **mixing cells with and without a species** in one cluster, which forces restoring cells that do not help.
- Making all 3k species a hard key may explode the number of strata, depending on how much the ranges overlap. **Measure, don't assume.**

Options, in increasing cost:

1. **Continuous feature.** Suitability (divide out area and `bio_S`; no per-species standardisation, since all are 0–100), PCA, or quantise with 0 as its own bin.
2. **Dual-weighted 1-D feature.** v_r = Σ π_s · a_{s,r}, with π from the previous year or a coarse solve. Non-binding species drop out automatically.
3. **Hard key only for binding or rare species.** Use rows with π ≠ 0, or `tight` / `straddle` rows from bound propagation, **after prescreen**: only rows that reach the solver (`*_MIN_AREA_HA` filtered, `active`, not dropped as redundant).
4. **Hashing for the hard key.** Two random int64 projections of each cell's 0/1 presence pattern → `pd.factorize`. This takes seconds.

Suitability changes by `target_year`, which fits the per-year re-clustering anyway.

### Error measurement

**A-posteriori bound (Zipkin).** From the aggregated LP's duals π, compute reduced costs on the original columns:

```
optimality loss ≤ Σ max(0, dᵢ) · (ubᵢ − xᵢ),   d = obj − Aᵀπ
```

This is one sparse product. It also says **which clusters to split**: columns with positive reduced cost want to leave the cluster's shared share.

**Adaptive refinement.** Coarse cluster (50–100×) → solve → bound → split the worst clusters → repeat until the bound is < 1 % (typically 3–5 rounds). Target *objective* error, not per-coefficient tolerance. This connects naturally with the existing `column_generation` branch.

**Expectations (unmeasured):** objective likely < 1 % loss at ~10× compression. Spatial pattern error is larger than the totals error; de-averaging recovers most of it.

### Relation to RESFACTOR

`RESFACTOR` is a fixed, data-blind clustering: spatial blocks, one representative cell, area × `RESMULT`. Its sampling bias needs ~110 corrections across the code (74 in `data.py`):

- `REAL_AREA × RESMULT` (with a TODO on actual area);
- `get_resfactored_lumap` / `LU_RESFACTOR_CELLS` (land-use proportions);
- stream length and density block sums, block-averaged eligibility;
- GBF3 targets pre-computed per `resfactor`;
- lost regional-adoption zones;
- raster transform rescaling.

Data-driven exact aggregation makes all of these unnecessary in principle.

**Caveat.** Table-level R·A·P saves **only the solve**. RES1 data loading, economics (~7 GB), `cell2col`, writing and reporting still run at RES1. To replace RESFACTOR for fast debugging, push the cell→cluster map down into `Data` as a `CLUSTER` mode: aggregate inputs once, treat clusters as cells downstream, and map outputs back with P. The hard part is that `Data` assumes a fixed cell set over the run, while clusters are re-formed each year.

### Added after the first test: keep blobs through the whole run

The first grouped solve (RES1, 2045 → 2050, one scenario) was about 10× faster on equal CPUs, with 0.0154 % objective loss. Grouping the tables saves only the solve, though: `Data` holds about 150 GB and the tables build to 211 GB. Two ideas extend the saving, neither tested.

**1. Decide blobs in table space, apply them on the data side.**

- The signature needs the tables, because exclusions, bounds, base and arcs do not exist yet in `Data`, and cells × (column type, row) is a matrix SVD handles directly. The memory, however, sits in `Data`.
- So: one offline RES1 table build per scenario, ideally signing cells across several years, saves `blob_of_cell`. At run time, `Data` reads the RES1 inputs, aggregates them to blobs and frees the cell arrays, so everything downstream treats a blob as a cell.
- **Blobs can stay fixed over the years, if they are never de-averaged:** cells with the same base taking the same shares have the same base next year. This removes the conflict between `Data`'s fixed cell set and yearly re-clustering.
- **Main risk:** blobs chosen for one year may fit worse later, as yields, climate and prices shift.
- **Cheapest test:** group another year's tables with the saved 2045 blobs, and solve it.
- **To measure first:** what makes up `Data`'s 150 GB.

**2. Keep blobs through write, and expand to RES1 only for maps.**

- Reassigning blobs to cells adds pressure on write, so write would work on blobs. A RES1 layer is then a lookup, `blob_values[blob_of_cell]`, at the point a map is needed.
- **Exact:** area, totals and every sum over a blob's cells, including state, NRM and water-region tables, since those regions are in the blob key.
- **Not exact:**
  - A per-hectare map (profit, yield, GHG, species) would show the blob's area-weighted average, so a cell's own value is lost. Share maps are exact.
  - Tables by a region outside the key (IBRA, CMA, or a new overlay) need the cell map, or those regions added to the key.
  - A blob spans all its cells, so a map can show the same value over cells far apart. Worth saying so in the report.
- **Decide:** write or report as the place to expand. The later it happens, the less write holds. Check which report layers are read cell by cell.

**3. The blob as the unit that holds every value.** Taken together, these make a blob the run's unit and a cell only a pointer to it:

```
blob_of_cell:  4.2M cells → blob id     (the only thing kept at RES1)
values:        one number per blob      (shares, cost, GHG, water, ...)
a cell's value = values[blob_of_cell[cell]]
```

- **In the code:** the `cell` dimension holds blobs, and its name can stay `cell`, so `write.py`'s dense multiplying and grouping run unchanged on arrays about 24× shorter (177,373 blobs against 4,218,733 cells at SVD 1.0). `data.NCELLS` becomes the blob count.
- **How `Data` aggregates:** totals and areas sum; per-hectare inputs are area-weighted averages. Each input needs the right rule. Check whatever assumes a real location (`COORD_LON_LAT`, the raster transform) and whatever is read directly at RES1 (the GBF8 layers).
- **Unlike RESFACTOR:** cells go together by similar coefficients, not as neighbours, and a value is the sum of all its cells, not one centre cell scaled.
- **What a blob holds:** totals and shares are exact, and per-hectare values are averages. An individual cell's own value cannot be recovered from the blob alone.
- **Splitting a blob's value back to cells.** All of these keep the blob's total. For example, 120 t GHG over 1,000 ha:
  - **By area (the default):** 0.12 t/ha, so a 150 ha cell gets 18 t. The report already stores per-hectare values, so it needs no extra step.
  - **Each cell's own coefficient × the blob's share (exact):** for overlay work and other cell-level questions. It needs the RES1 inputs read once, but no solve.
  - **Not an equal split:** 120 / 10 = 12 t per cell overstates small cells and understates large ones.
- **For results shared with others:** keep the greedy reassignment of fractional blobs (no solve). A fractional blob spreads a small change thinly over all its cells, where the full solve would pick the best ones, and an overlay would read that spread as real. Measure overlay accuracy on zones outside the blob key (IBRA, SA2) before quoting it.
- **Sparse arrays do not replace this.** They help the dvars (2045: 4.24 M nonzero land entries against 236 M dense), but the coefficients are mostly filled and xarray's sparse support is limited. Use blobs first, and add sparse only if needed.

**4. The dashboard map from blobs.** The report and the dashboard can use the same lookup.

- **Report:** `create_report_layers.py` places values through `index_merc` (pixel → cell). `blob_of_cell[index_merc]` makes it pixel → blob, so nothing expands to 4.2 M cells, and pixels still line up with real cells at RES1. The colour-bar limits, now scaled by `RESFACTOR`, need real areas instead.
- **Browser:** `regions_map.js` already decodes each layer to pixels and colours them in a loop. So a layer can be a small array of blob values, with one `blobIndex` raster loaded once per run: `values[blobIndex[pixel]]`.
- **Size:** a dense layer falls from about 20 MB per year to about 0.95 MB before compression, about 20×. A RES1 report's `map_layers` reached 59 GB across 6,819 files. Sparse layers are already about 0.9 MB through deflate, so blob values must be deflated too (`DecompressionStream`, built into browsers). `blobIndex` size is to be measured.
- **The download button still works:** it already builds its TIFF in the browser with `buildGeoTiff`, so it would fill the pixels from `values[blobIndex]`.
- **Label the output:** a per-hectare TIFF holds blob averages, and it gets used outside the dashboard, where a pixel could be read as that cell's own value. State it in the file name or metadata, and in the dashboard.
- **Look at one first:** render a per-hectare map from real blobs. Scattered blobs may look speckled.

**Order:** fixed blobs must hold up across years first; all of the above depends on them.

**5. GBF8 with many species: keep blobs reasonable at RES1.** For example, 1,000 species targets. Nothing here is tested.

- **Blobs alone do not solve it; the rows are the problem.** The model without species has 331 wide rows and about 21.8 M nonzeros at about 118,000 blobs. 1,000 more rows, each touching a large share of about 5.5 M columns, could reach billions of nonzeros at the rough upper limit. So pair blobs with `BIO_ROW_GENERATION`: blobs shrink the columns, and row generation adds rows only for species that fall short.
- **Options for grouping:**
  - **Species count per cell, in buckets:** one new key level, so few extra blobs. It loses which species a cell holds: two cells with 50 species can differ in the rare one. The blob must then restore both to reach that species' target, and one does nothing for it. Rows stay exact, so this costs accuracy, not validity. Simple, so use it as the baseline.
  - **Species in the SVD signature (start here):** GBF8 rows join the signature automatically, as SNES and ECNES do, and cells with similar species mixes group. Rare species may get lost.
  - **Dual-weighted value:** v_r = Σ π_s · a_{s,r}, one number per cell. Non-binding species drop out. Needs duals from an earlier solve.
  - **Exact key for rare or binding species:** added to any of the above, only where the error asks for it.
- **Test:**
  1. Count blobs for 1,000 species under count buckets and under SVD, with no solve.
  2. Measure mixed presence: how often a blob holds cells with and without a given species.
  3. Run a grouped solve with row generation. A full RES1 reference with GBF8 may not exist, so a smaller region may have to serve.
- **Layers change by year,** so blobs chosen for one year may not fit another. This bears on fixed blobs.

### Open checks

- Structural rows: confirm none are row-scaled before summing them in R (or restore `scale` first).
- Any future spatial-adjacency constraint (for example connectivity) would break the "linear sum over cells" assumption.
- Barrier runs must use crossover for both the aggregated LP and the polishing LP.

### Roadmap

| Step | Work | Effort | Decision it informs |
|---|---|---|---|
| 1 | **Compression-rate curves** on existing RES1 tables (no solve): clusters vs tolerance (0.5–5 %, before / after PCA); strata count with species hashing (all 3k / tight-straddle / π ≠ 0); quantisation levels (1–3 bins) | ~½ day | Is it worth it? Can 3k species be a hard key? |
| 2 | **Single-year prototype** `aggregate.py`: strata, P / R, R·A·P, map-back, Zipkin bound, numpy de-averaging, polishing fallback; hook into `step.py`. Compare **full RES1 vs RES1-clustered (~25×) vs RES5** on one state (or the NECMA / GBCMA area): objective, land-use areas, maps, `bound_report` | days | Does clustering beat RESFACTOR at equal size? |
| 3 | **Multi-year** 2010–2050 with yearly re-stratification | — | Recursion stability |
| 4 | **`CLUSTER` mode in `Data`**, run alongside RESFACTOR | — | Build-time savings |
| 5 | Remove RESFACTOR corrections and the per-`resfactor` GBF3 target table | — | Only after 4 is validated (RESFACTOR is still the fastest debug path and may be in use at CSIRO) |

**Suggested first version:** leave species out (or use GBF2 / GBF3 only). Hard strata = base (m, lu) + masks + regions. Features = profit, GHG, water, yield per ha, quantised. Then add GBF8 as a dual-weighted feature, and promote species to hard keys only where the error bound says so.

**Possible publication:** data-driven exact aggregation for large-scale land-use LPs, with vertex-based de-averaging and multi-year recursion. It is methodologically novel relative to uniform resolution coarsening.

---

## 20261102 — `write.py` refactor: obey the memory cap, survive a failed year, remove remnants, read flat

*Line numbers are for `luto/tools/write.py` at commit `2b7a96f5`. The review was read-only: nothing below has been changed.*

### TL;DR

- The memory cap is a prediction, not a limit, and runs exceed it.
- `max_cell_magnitudes.json` is written only after every task succeeds, so one failed year loses all of it.
- The review also found outdated docstrings, dead code, and bugs that put wrong numbers in the outputs (listed at the end).

### 1. Make write obey the memory cap

**Problem.** Workers are sized from RES5 per-function peaks scaled to the run's resolution. With a 250 GB cap, write still exceeds the cap, so the design cannot be relied on. An older data point: the June RES1 G0005 redo-write used 865 GB under PBS with `WRITE_REPORT_MAX_MEM_MB` at 456 GB in its settings. That figure covers write and report together on older code, so it is an indication, not a measurement of the current design.

**How it works now.** `n_jobs = min(32, budget // (peak_mb + 500))`, with `budget = max(4096, WRITE_REPORT_MAX_MEM_MB − parent RSS)` and `peak_mb = peak_mb_RES5 × (5 / RESFACTOR)²` (lines 71–99, 461–469).

**Why it cannot hold:**

- **The profile is stale.** It dates from 2026-05-30 (line 72), before the later refactors.
- **Some entries look measured with the feature off.** Both GBF8 functions show 1 MB and 1 s (lines 90–91), and `GBF8_TARGET` defaults to `'off'`. They would get the most workers, yet `write_biodiversity_GBF8_scores_species` builds species × NCELLS eagerly (line 5103).
- **Peak is not proportional to cells alone.** It also grows with:
  - species and layers (the GBF3 / GBF4 frames and batches);
  - biodiversity backends (`write_biodiversity_quality_scores`);
  - simulated periods (`period_flows` in `write_area_transition_start_end`).
- **Thread count is uncontrolled.** `write.py` sets no dask scheduler, so each worker runs dask's default thread pool, and peak depends on threads × chunk size.
- **Nothing enforces the cap.**
  - The estimate is made once per tier and never checked against actual use.
  - The `max(4096, ...)` floor and `max(1, ...)` still run a task when the budget is already gone.
  - A single task larger than the budget just runs.
- **The report stage is outside the budget.** `create_report` runs after `write_data` with no such sizing.

**Plan.**

1. Replace the prediction with **admission control**. The parent starts the next task only when the process tree's measured memory plus that task's estimate fits under the cap. Otherwise it waits for a task to finish.
2. Learn the estimates during the run. Record each task's observed peak, so the first year calibrates the later ones. Start conservatively, with one task at a time for an unknown function.
3. Bound the working set inside the heavy functions: set the dask scheduler and thread count explicitly, and batch the species and layer loops by bytes rather than by a fixed count. A cap cannot help if one task alone exceeds it.
4. Free what the parent holds where possible. Workers already share `data` through joblib's memmap.
5. Decide whether the cap also covers `create_report`.

**Done when:** a RES1 write under a 250 GB cap peaks at or below 250 GB by PBS accounting, and a RES5 write with a deliberately low cap stays under it by running slower rather than being killed.

**Also fix:**

- The comment on `WRITE_REPORT_MAX_MEM_MB` in `settings.py` still describes the old "RESFACTOR = 13" estimate.
- `CLAUDE.md` still calls the setting `WRITE_REPORT_MAX_MEM_GB`.

### 2. Keep magnitudes when a year fails

**Problem.** Tasks return their magnitudes to the parent, which accumulates them in the module-level `MAX_CELL_MAGNITUDE` and writes `max_cell_magnitudes.json` only at the end of `write_data` (line 508). `create_report_layers.py:589` needs that file, so one failed task means rerunning every write.

**Proposed idea:** start the maximum at 0 and replace it as each larger value arrives, so a usable file exists at all times.

**Recommended form: one small file per finished task**, `out_<year>/magnitudes/<write_function>.json`. The worker writes it atomically (temp file, then `os.replace`) as its last step, and `save_report_layer` merges all of them at the start. Compared with one running max:

- A rerun replaces that task's own numbers. A running max cannot shrink after a wrong value from a failed or changed run.
- Parallel workers never contend for one file.
- The file doubles as a "this task finished" marker, so a resumed write can skip finished tasks, which is what actually avoids the rerun.
- The parent stops accumulating lists in memory.

**Details to keep:**

- Store both ends, `[min, max]`. An initial 0 would change the colour bar: all-positive layers would start from 0 instead of their real minimum, and all-negative layers would end at 0. Start from empty, not 0.
- `get_mag` takes quantiles, while water and the start-to-end transition take true min / max. The merge's min-of-mins and max-of-maxes matches what `create_report_layers.py` does now.
- Keep the NaN → 0 cleaning, which `write_data` applies at line 507.
- Update `docs/CLAUDE_OUTPUT.md` § "Colorbar Range: MAX_CELL_MAGNITUDE Design" and `docs/CLAUDE_SKILL/redo_failed_write.md`.

### 3. Remove the temp-dir remnants

The design that saved species layers to a temp dir and concatenated them has been replaced by per-batch chunk directories.

| Item | State | Action |
|---|---|---|
| `concat_tmp2nc` (line 181) | No callers in `luto/` | Remove |
| `import glob` | Used only by `concat_tmp2nc` | Remove with it |
| Mention of `concat_tmp2nc` in the `save2chunk` docstring (line 165) | Stale | Reword |
| `tmp_dir` parameter of `_save_fullsize_layers` (lines 4222, 4254) | Stale name; it is the chunk dir | Rename |
| "Set up region levels and tmp dirs" comments (lines 4115, 4582) | Stale | Reword |
| `*_chunks` dirs, `manifest.json`, `save2chunk` | **Live**: read by `create_report_layers.py` and `create_report_data.py` | Keep |

### 4. Update docstrings to the current design

Each of these contradicts what the code does now:

| Line | Says | Code does |
|---|---|---|
| 551 | `n_jobs = floor(WRITE_MEM_BUDGET_MB / peak_delta_mb)` | That setting does not exist; the formula is at 468–469 |
| 168 | Magnitude "computed before writing" | `get_mag` runs after the write (176) |
| 248–250 | Chunks "stay near target_mb" | Sized before later broadcasts multiply them |
| 48 | `ag_quantity` "imported but skip" | It is used |
| 598 | Mosaic "reflects LP allocation, not existing installations" | Argmax (631) runs after the injection (615) |
| 812 | "No dvar file exists" | Line 665 writes it every year |
| 867, 3633, 4055, 4524 | `-> np.ndarray` / `-> None` | They return tuples |
| 1133, 1136 | "Can be freed"; "never builds full 4D" | The lazy graph keeps it alive; 1143–1146 build it eagerly |
| 1039 | "None values" | It filters NaN |
| 1674–1676 | "(am, lm, lu, cell)" | (am, cell) |
| 1882 | "1 t CO2e" | The threshold is 1e-3 |
| 3558 | Writes "to VUE_modules" | Writes to `data.path` |
| 3605 | Mentions a mosaic | There is none |
| 3671, 4079 | "No chunking"; "Unify chunk" | 3671 calls `chunk_unify_size`; SNES has no chunking |
| 4504, 4869 | "Magnitudes from chunk dirs" | They accumulate in the loop |
| 5037–5038, 5235–5236 | "Skip writing" | The am branch writes an ALL fallback (5057, 5255) |

`write_data`, `write_settings`, `create_report` and `save_csv` have no docstring.

### 5. Readability: keep functions flat

**Principle:** long plain top-to-bottom sections, few extracted helpers. A helper stays only if it is truly shared and its name saves reading.

**Inline:**

- `_save_layers` (both transition writers)
- `_make_exist_cost_da`, `_expand_exist`
- `_minmax`
- ECNES's `_score_df` and `_valid_layers`, which are redefined on every loop pass (lines 4645, 4677)

**Keep:** `_flow_report`, `_chained_blocks`, and the shared top-level `save2nc`, `save2chunk`, `get_mag`, `add_all`.

**Make consistent:**

- `to_region_and_aus_df` and `bio_to_region_and_aus_df` are the same loop (lines 264, 300). One function is enough.
- The GBF3 / SNES / ECNES tails (lines 3889–4033, 4370–4502, 4736–4867) are nearly line-for-line copies. This is the one new shared helper that looks justified.
- SNES differs from the others without an evident reason: it keeps `Type='ALL'` in its sum chunks (4342 vs 3877, 4722) and uses layer order `lm, lu, species` rather than `species, lm, lu`. NVIS keeps `Target_by_Percent` where the others rename it.
- `write_transition_nonag2ag` follows a different design from the two ag writers: NRM only, no `region_level`, its own chunking. Cost thresholds also differ (1e4 at 1859, 1000 at 2087).
- Five identical blocks at 1482–1506; a branch that differs only by arguments at 2875–2900; repeated blocks at 2913–2933 and 3024–3044.
- `combinations` is imported locally three times.
- Rename single-letter variables (`D, q, g, c, v, n, A, s, e, t, l, P, f`). Only `m, r, j, k` stay, with `x` for the LP solution.

**Dead code, for the owner to decide on deletion:**

- `process_chunks` (338): no callers
- `nonag2ag_cost_jms_AUS` (1187)
- `nonag_as_dry` (3075)
- `ghg_df_AUS` alias (2662)
- `non_ag_quantity` import (56)
- `yr_cal_sim_pre=` parameters (1703, 1930, 2149): no caller passes them
- `add_all` at 2961, 2978: ALL is dropped by the inner join
- `.replace` at 3030, 3037, 3044: no-op on names
- Renames with no matching column at 2546, 3048, 3523
- Empty Bio section (1921–1925)

### Bugs found during the review (not fixed)

Fix these first, or separately from the refactor, so that output changes are not mixed with code moves. None has been confirmed against a run's output files yet.

**Confirmed by reading the code:**

| Line | Output affected | Cause |
|---|---|---|
| 2992–2995 | `Water yield inside LUTO` and `Water Net Yield` in the watershed table ×2 | The filter removes `lm` and `am` ALL rows but not `lu` ALL rows, from frames built with `add_all`. This can make a missed water limit look met in the report; the solver is not affected |
| 3612–3614 | `xr_biodiversity_GBF2_priority_sum` map inflated: ag ×4, non-ag ×2, am ×8 | Sums over `lm` / `lu` / `am` after `add_all` (3455–3457) already added ALL slices |
| 1262, 1533 | Sum Profit omits renewables | `raw_profit_am_pre` uses the matrices zeroed at 1258–1260 and is never updated, while `xr_profit_am` includes renewables |
| 3475–3511 | GBF2 pseudo rows lack `region_level` | The same defect `a05cd4c6` fixed for quality. The report groups on this column; no crash has been reproduced |

**From the review only, still to check:**

- **Outside-LUTO climate-change impact (2950, 2962):** always zero when `WATER_CLIMATE_CHANGE_IMPACT = 'on'`. Both lines select the base year, while `water.py:532` uses `yr_cal`. The intent is unverified.
- **Switch crosstab (2563):** subtraction without `fill_value` turns a land use present on only one side into NaN, and the query then removes it.
- **National totals (283→288, 322→328, 1794→1833):** AUS rows sum regions already filtered at |v| > 1, so they are slightly low and later 1e-3 GHG filters have no effect.
- **Signed filters (3830, 3846, 3862, 4217, 4681):** a signed `> 1` drops negative-score layers, while `to_region_and_aus_df` uses `abs`.
- **Mask file (3535):** check-then-write across workers on the same file.
