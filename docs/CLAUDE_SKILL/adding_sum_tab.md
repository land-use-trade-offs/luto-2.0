# Skill: Adding a "Sum" Tab (Ag + Am + NonAg)

This skill documents the process of adding a "Sum" tab that aggregates data across Ag, Am, and NonAg categories. Applied to both **Production** and **Economics** modules.

## Overview

The Sum tab sums values from all three categories (Ag, Am, NonAg) into a single view. The selection hierarchy is **Category → Water → Landuse/Commodity**, with no MapType or AgMgt levels.

**Key design decisions:**
- `lu='ALL'`/`Commodity='ALL'` and `lm='ALL'` use actual sums (not lumap mosaic) for Economics; Production uses a **mosaic for `Commodity='ALL'`** and float for per-commodity layers
- NonAg land uses are **appended** to Ag land uses (they are different sets), not merged
- NonAg is assigned to `lm='dry'` (NonAg has no water/irrigation dimension)
- Am is summed over the `am` dimension before combining with Ag
- Am `'ALL'` rows must be **excluded before summing** to avoid double-counting

## Files Modified (6 files per module)

### Step 1: `luto/tools/write.py` — Upstream data generation

**What:** Add a new section that computes the summed data and saves it as a NetCDF file.

#### Pattern A: Economics Sum (all-float layers)

```python
# --- Section 4: Sum profit (Ag + Am + NonAg) ---

# Get raw (pre-ALL) profits from each category
raw_profit_ag = (ag_dvar_mrj * profit_ag).drop_vars('region')           # (lm, lu, cell)
raw_profit_am_pre = (am_dvar_mrj * (am_revenue - (am_cost + am_trans))) # (am, lm, lu, cell)
am_sum_profit = raw_profit_am_pre.sel(lm=['dry', 'irr']).sum('am').drop_vars('region')  # (lm, lu, cell)

raw_profit_nonag = (non_ag_dvar * non_ag_profit_mat).drop_vars('region')  # (lu, cell)
# Assign nonag to lm='dry', fill irr with 0
nonag_as_dry = raw_profit_nonag.expand_dims('lm').assign_coords(lm=['dry']) \
    .reindex(lm=['dry', 'irr'], fill_value=0)

# Combine: ag+am for ag land uses, nonag for nonag land uses
ag_lus = list(data.AGRICULTURAL_LANDUSES)
ag_plus_am = raw_profit_ag.sel(lm=['dry', 'irr']) + am_sum_profit.reindex(lu=ag_lus, fill_value=0)
sum_dry_irr = xr.concat([ag_plus_am, nonag_as_dry], dim='lu')

# Add ALL lm aggregate (sum over water types)
sum_all_lm = sum_dry_irr.sum('lm', keepdims=True).assign_coords(lm=['ALL'])
sum_profit = xr.concat([sum_all_lm, sum_dry_irr], dim='lm')

# Add ALL lu aggregate (sum over all land uses)
sum_all_lu = sum_profit.sum('lu', keepdims=True).assign_coords(lu=['ALL'])
sum_profit = xr.concat([sum_all_lu, sum_profit], dim='lu')

# Stack and save — ALL layers are float (no mosaic needed)
sum_profit_stack = sum_profit.stack(layer=['lm', 'lu']).compute()
_save2nc(sum_profit_stack, os.path.join(path, f'xr_economics_sum_profit_{yr_cal}.nc'))
```

**Key points (Economics):**
- Use pre-ALL variables (before `add_all` is called) so we control the aggregation
- Both `lm='ALL'` and `lu='ALL'` are actual sums, not categorical mosaics
- Add magnitude entry: `'Economics_sum': {'sum_profit': _get_mag(sum_profit_stack)}`

#### Pattern B: Production Sum (mosaic for ALL + float per-commodity)

Production differs from Economics: `Commodity='ALL'` uses a **categorical lumap mosaic** (same as Ag), while per-commodity layers are float sums.

```python
# --- Sum (Ag + Am + NonAg): stack and save ---
# Exclude 'ALL' from am to avoid double-counting, then sum over 'am'
am_sum_mrc = am_p_amrc.sel(
    lm=['dry', 'irr'],
    Commodity=[c for c in am_p_amrc.coords['Commodity'].values if c != 'ALL']
).sum('am')
non_ag_as_dry = non_ag_p_rc.expand_dims('lm').assign_coords(lm=['dry']) \
    .reindex(lm=['dry', 'irr'], fill_value=0)

sum_dry_irr = (ag_q_mrc.sel(lm=['dry', 'irr'])
               + am_sum_mrc.reindex_like(ag_q_mrc.sel(lm=['dry', 'irr']), fill_value=0)
               + non_ag_as_dry.reindex_like(ag_q_mrc.sel(lm=['dry', 'irr']), fill_value=0))

sum_all = sum_dry_irr.sum('lm', keepdims=True).assign_coords(lm=['ALL'])
sum_mrc = xr.concat([sum_all, sum_dry_irr], dim='lm')

# Float layers: per-commodity production sums
sum_mrc_stack = sum_mrc.stack(layer=['lm', 'Commodity'])

# Mosaic layers: use xr_map_lumap for Commodity='ALL' (categorical land-use map with lm splits)
lumap_mosaic = cfxr.decode_compress_to_multi_index(
    xr.load_dataset(os.path.join(path, f'xr_map_lumap_{yr_cal}.nc')), 'layer')['data'].unstack('layer')
sum_mosaic = lumap_mosaic.expand_dims('Commodity').assign_coords(Commodity=['ALL'])
sum_mosaic_stack = sum_mosaic.stack(layer=['lm', 'Commodity'])

# Concat mosaic (ALL) + float (per-commodity), then save
sum_cat_stack = xr.concat([sum_mosaic_stack, sum_mrc_stack], dim='layer').drop_vars('region').compute()
_save2nc(sum_cat_stack, os.path.join(path, f'xr_quantities_sum_{yr_cal}.nc'))
```

**Key differences from Economics:**
- Uses `reindex_like()` instead of manual `reindex(lu=...)` since commodities align across sources
- `Commodity='ALL'` uses **categorical mosaic** (loaded from `xr_map_lumap`), not a sum
- Must `xr.concat([sum_mosaic_stack, sum_mrc_stack])` to combine mosaic + float layers
- Production magnitudes include sum: `'sum': {cm: _get_mag(sum_mrc.sel(Commodity=cm)) for cm in data.COMMODITIES}`
- The global min/max includes sum values: `vals = [*ag, *non_ag, *am, *sum]`

### Step 2: `luto/tools/write.py` — Update call site

In `write_output_single_year()`, if merging functions (as done for Economics), update the `delayed()` call to use the merged function. Otherwise ensure the new Sum section runs within an existing function that has access to all three categories' variables.

### Step 3: `luto/tools/report/create_report_data.py` — Chart data generation

**What:** Add a section that loads/combines data from all three categories and outputs a chart JS file.

#### Economics chart data pattern

```python
# ---- Economics Sum ----
profit_ag_df = pd.read_csv(files_ag_profit)   # filter out ALL rows
profit_am_df = pd.read_csv(files_am_profit)   # filter out ALL, sum over Management Type
profit_na_df = pd.read_csv(files_na_profit)   # filter out ALL, assign Water_supply='Dryland'

econ_sum = pd.concat([profit_ag_df, profit_am_df, profit_na_df], ignore_index=True)

# Add ALL water aggregate
econ_sum_all_water = econ_sum.groupby(['region', 'Land-use', 'Year'])[['Value ($)']].sum() \
    .reset_index().assign(Water_supply='ALL')
econ_sum = pd.concat([econ_sum_all_water, econ_sum], ignore_index=True)

# Output hierarchy: region -> water -> [series by landuse]
```

**Output file:** `Economics_Sum.js` with hierarchy `region → water → [series(name=landuse)]`

#### Production chart data pattern

Production reuses already-loaded DataFrames (`quantity_ag_non_all`, `quantity_am_non_all`, `quantity_non_ag`):

```python
# ---- Sum production (Ag + Am + NonAg) ----
quantity_non_ag_with_water = quantity_non_ag.copy()
quantity_non_ag_with_water['Water_supply'] = 'Dryland'

quantity_sum = pd.concat([
    quantity_ag_non_all[['region', 'Water_supply', 'Commodity', 'Year', 'Production (t/KL)']],
    quantity_am_non_all[['region', 'Water_supply', 'Commodity', 'Year', 'Production (t/KL)']],
    quantity_non_ag_with_water[['region', 'Water_supply', 'Commodity', 'Year', 'Production (t/KL)']],
], ignore_index=True)

# Add ALL water level
quantity_sum_all_water = quantity_sum\
    .groupby(['region', 'Commodity', 'Year'])[['Production (t/KL)']]\
    .sum(numeric_only=True).reset_index().assign(Water_supply='ALL')
quantity_sum = pd.concat([quantity_sum_all_water, quantity_sum], ignore_index=True)

# Group → time series, then nest: region → water → [series(name=Commodity)]
```

**Output file:** `Production_Sum.js` with hierarchy `region → water → [series(name=Commodity)]`

**Key:** Both modules use `_non_all` (pre-filtered) DataFrames to avoid double-counting ALL rows.

### Step 4: `luto/tools/report/create_report_layers.py` — Map layer generation

**What:** Add a call to `get_map2json()` for the Sum NetCDF file.

#### Economics (all-float, no categorical legend)

```python
economic_magnitudes_sum = cell_magnitudes.get('Economics_sum', {}).get('sum_profit', [0.0, 0.0])
economic_min_max_sum = (min(economic_magnitudes_sum), max(economic_magnitudes_sum))
profit_sum = files.query('base_name == "xr_economics_sum_profit"')
get_map2json(profit_sum, None, None, legend_float, economic_min_max_sum,
             f'{SAVE_DIR}/map_layers/map_economics_Sum_profit.js')
```

**Key:** `legend_int=None, legend_int_level=None` because ALL layers are float (no categorical mosaic).

#### Production (has categorical ALL layer)

```python
quantities_sum = files_quantities.query('base_name == "xr_quantities_sum"')
get_map2json(quantities_sum, legend_ag, {'Commodity':'ALL'}, legend_float, prod_min_max,
             f'{SAVE_DIR}/map_layers/map_quantities_Sum.js')
```

**Key:** Uses `legend_ag` and `{'Commodity':'ALL'}` because `Commodity='ALL'` is a **categorical mosaic** (lumap), same as Ag. Per-commodity layers use `legend_float`.

### Step 5: `luto/tools/report/VUE_modules/services/MapService.js` — Register map data

Add a `'Sum'` entry in the module section. **Place it first** (before Ag):

```javascript
// Economics example (has sub-types per mapType)
'Sum': {
  'Profit': {
    'path': 'data/map_layers/map_economics_Sum_profit.js',
    'name': 'map_economics_Sum_profit'
  },
},

// Production example (single entry, no mapType sub-level)
'Sum': { 'path': 'data/map_layers/map_quantities_Sum.js', 'name': 'map_quantities_Sum' },
```

### Step 6: `luto/tools/report/VUE_modules/services/ChartService.js` — Register chart data

```javascript
// Economics
"Sum": {
  "path": "data/Economics_Sum.js",
  "name": "Economics_Sum",
},

// Production
"Sum": {
  "path": "data/Production_Sum.js",
  "name": "Production_Sum",
},
```

### Step 7: `luto/tools/report/VUE_modules/views/<Module>.js` — Vue view component

**What:** Add Sum to the category list, wire up selection cascade + data access, and add `loadScript` calls.

**Changes checklist:**
1. Add `"Sum"` to `availableCategories` array (**first position**)
2. Add `previousSelections` entry: `"Sum": { water: "", landuse: "" }` (no mapType/agMgt)
3. Add `loadScript` calls in `onMounted` for both map and chart Sum data
4. Add Sum branch in `selectMapData` computed: `mapData[water][landuse][year]`
5. Add Sum branch in `selectChartData` computed: `chartData[water]` filtered by landuse
6. Add Sum branch in category watcher — save previous, cascade water → landuse
7. Add Sum branch in water watcher — cascade landuse
8. Add Sum branch in landuse/commodity watcher — save selection
9. Show Water row for Sum (same as Ag): update `v-if` to include `'Sum'`
10. Hide MapType/AgMgt rows when Sum selected

**Selection hierarchy for Sum:**
```
Category: [Sum, Ag, Ag Mgt, Non-Ag]
    └─ Water: [ALL, Dryland, Irrigated]
        └─ Landuse: [ALL, Beef, Sheep, ..., Environmental Plantings, ...]
```

**Map data access:** `mapData[water][landuse][year]` (same structure as Ag)
**Chart data access:** `chartData[water]` → filter series by landuse (same structure as Ag)

## Cross-Category Selection Persistence

This commit also introduced a **cross-category selection fallback** pattern applied to **all** Vue views (Area, Biodiversity, Economics, GHG, Production, Water). This ensures switching categories preserves selections when possible.

### The pattern

When restoring a previous selection, use `|| currentValue` as fallback:

```javascript
// BEFORE (only restores category-specific previous selection)
const prevWater = previousSelections.value["Ag"].water;
selectWater.value = (prevWater && availableWater.value.includes(prevWater))
    ? prevWater : (availableWater.value[0] || '');

// AFTER (falls back to current selection if no category-specific one saved)
const curWater = selectWater.value;  // capture BEFORE switching
const prevWater = previousSelections.value["Ag"].water || curWater;
selectWater.value = (prevWater && availableWater.value.includes(prevWater))
    ? prevWater : (availableWater.value[0] || '');
```

### Implementation steps
1. At the **top** of the category watcher (after saving old selections), capture current values:
   ```javascript
   const curWater = selectWater.value;
   const curLanduse = selectLanduse.value;  // or selectCommodity.value for Production
   const curAgMgt = selectAgMgt.value;
   ```
2. In **every** `previousSelections.value[cat].xxx` lookup, add `|| curXxx` fallback
3. Apply to **all** watchers that restore selections (category, water, agMgt watchers)
4. Apply to **all** categories in the view (Ag, Ag Mgt, Non-Ag, Sum)

**Why:** Without this, switching from Ag→Non-Ag→Ag loses the selection because Non-Ag doesn't save water/agMgt values. The `|| currentValue` fallback tries to keep whatever the user last saw.

## Dimension Handling Summary

| Source | Raw dims | Transform for Sum |
|--------|----------|-------------------|
| Ag | `(lm, lu, cell)` | Use directly (select `lm=['dry','irr']`) |
| Am | `(am, lm, lu, cell)` | Exclude `'ALL'` from am/Commodity, then `.sum('am')` |
| NonAg | `(lu, cell)` | `.expand_dims('lm').assign_coords(lm=['dry'])` then `.reindex(lm=['dry','irr'], fill_value=0)` |

**Combining:** `xr.concat([ag_plus_am, nonag_as_dry], dim='lu')` — appends nonag LUs after ag LUs.

**ALL aggregates:**
- `lm='ALL'`: `.sum('lm', keepdims=True).assign_coords(lm=['ALL'])`
- `lu='ALL'` / `Commodity='ALL'`:
  - **Economics**: `.sum('lu', keepdims=True)` — real sum (all-float)
  - **Production**: **categorical mosaic** from `xr_map_lumap` — NOT a sum

## Economics vs Production: Key Differences

| Aspect | Economics Sum | Production Sum |
|--------|-------------|----------------|
| Main dimension | `lu` (land use) | `Commodity` |
| `ALL` layer type | Float sum | Categorical mosaic (lumap) |
| `legend_int` in layers | `None` | `legend_ag` |
| `legend_int_level` | `None` | `{'Commodity':'ALL'}` |
| Magnitude key | `'Economics_sum'` | `'sum'` (inside `prod_magnitudes`) |
| Vue select variable | `selectLanduse` | `selectCommodity` |
| MapService structure | Nested by mapType: `Sum.Profit.path` | Flat: `Sum.path` |
