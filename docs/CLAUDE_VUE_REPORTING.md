# LUTO2 Vue.js Reporting System

Vue.js 3 dashboard (`DATA_REPORT/REPORT_HTML/index.html`) that renders LUTO simulation outputs as interactive maps and charts. No build step — all JS files are loaded via `<script>` tags from local `data/` directories.

## File Structure

| Path | Purpose |
|------|---------|
| `VUE_modules/views/` | View components (one per module) |
| `VUE_modules/services/MapService.js` | Map file registry: `module → category → mapType → {indexPath, indexName, layerPrefix}` |
| `VUE_modules/services/ChartService.js` | Chart file registry: `module → category → {path, name}` |
| `VUE_modules/data/map_layers/` | Pre-rendered base64 PNG map tiles — split into per-combo `.js` files + one `__index.js` per variable |
| `VUE_modules/data/` | Chart time-series data (`.js` files) |
| `VUE_modules/routes/route.js` | Vue Router configuration |

---

## xarray → Base64 Map Pipeline

How simulation outputs (`write.py` NetCDF) become the base64 PNG tiles rendered in Leaflet.

### Pipeline overview

```
write.py: xarray (1D cells) → stack dims → valid-layer filter → NetCDF
    ↓
create_report_layers.py: decode MultiIndex → iterate layers → rename/reorder
    ↓ per layer
map2base64(): 1D → 2D raster → reproject EPSG:3857 → RGBA render → PNG base64
    ↓
tuple_dict_to_nested(): flat tuple keys → nested dict
    ↓
window["map_name"] = { Water: { Source: { LU: { Year: {img_str, bounds, ...} } } } };
```

### Step 1 — Stack dims and save NetCDF (`write.py`)

```python
xr_stacked = xr_data.stack(layer=['lm', 'source', 'lu']).sel(layer=valid_layers)
_save2nc(xr_stacked, f'xr_economics_ag_cost_{yr_cal}.nc')
```

- All dims except `cell` are compressed into a `layer` MultiIndex (via `cf_xarray`).
- Only layers with non-zero AUS aggregates are kept (`valid_layers`), reducing file size.
- **Stacking order defines the JSON hierarchy.** Canonical order enforced by `rename_reorder_hierarchy` (see below).

### Step 2 — Decode and iterate (`create_report_layers.py → get_map2json`)

```python
xr_arr = cfxr.decode_compress_to_multi_index(xr.open_dataset(path), 'layer')['data']
for sel in xr_arr['layer'].to_index().to_frame().to_dict(orient='records'):
    # sel = {'lm': 'ALL', 'source': 'Area cost', 'lu': 'Apples'}
    arr_sel = xr_arr.sel(**sel)              # 1D cell array
    sel_rename = rename_reorder_hierarchy(sel)
    hierarchy_tp = tuple(sel_rename.values()) + (year,)
    # → ('ALL', 'Area cost', 'Apples', 2020)
```

### Step 3 — Canonical hierarchy order (`data_tools/__init__.py → rename_reorder_hierarchy`)

Enforces a fixed nesting order and human-readable labels for all map files:

```
1. am   → renamed via RENAME_AM_NON_AG
2. lm   → 'dry' → 'Dryland', 'irr' → 'Irrigated', 'ALL' → 'ALL'
3. other dims (source, type, commodity, …)
4. lu / from_lu  ← always last (land-use is the bottom-level UI selection)
```

`from_lu` (used in transition ag2nonag) is treated identically to `lu` — placed last.

### Step 4 — 1D → 2D raster remap (`map2base64`)

```python
# Template: 2D grid, each cell's position stored as its index value; negatives = outside LUTO area
np.place(rxr_arr.data, rxr_arr.data >= 0, arr_sel.values)
rxr_arr = xr.where(rxr_arr < 0, np.nan, rxr_arr)   # outside → transparent
```

### Step 5 — Reproject to Web Mercator

```python
rxr_arr = rxr_arr.rio.write_crs(rxr_crs)            # native CRS (GDA2020 / EPSG:7844)
bbox    = rxr_arr.rio.bounds()                       # capture lat/lon bounds before reproject
rxr_arr = rxr_arr.rio.reproject('EPSG:3857')         # Leaflet uses Web Mercator
```

Bounds stored as `[[lat_min, lon_min], [lat_max, lon_max]]` for Leaflet `imageOverlay`.

### Step 6 — Integer vs float rendering

| Layer type | Examples | Rendering |
|------------|---------|-----------|
| **Integer** | DVAR, lumap | Pixel = land-use code → RGBA lookup from `COLOR_AG/AM/NON_AG` |
| **Float** | Economics, GHG, Water, Biodiversity | Clip 1st–99th percentile, normalize 0–100, map through `COLORS_FLOAT_POSITIVE`; store raw `min_max` for colorbar |

### Step 7 — PNG → base64

```python
arr_4band = np.zeros((H, W, 4), dtype='uint8')
for code, rgba in color_dict.items():
    arr_4band[rxr_arr == code] = rgba
img_str = 'data:image/png;base64,' + base64.b64encode(
    Image.fromarray(arr_4band, 'RGBA').save(buf, 'PNG') or buf.getvalue()
).decode()
```

### Step 8 — Write split `.js` files

```python
_write_split_by_combo(flat_output, dim_names, save_path)
# For each unique (dim1, …, dimN) combo, writes:
#   map_area_Ag__Dryland__Apples.js  → window["map_area_Ag__Dryland__Apples"] = { "2020": {...}, "2025": {...} }
# Plus one index file:
#   map_area_Ag__index.js            → window["map_area_Ag__index"] = { dims: ["lm","lu"], tree: { "Dryland": ["Apples",...], ... } }
```

- `safe_key(s)` converts any string to a safe filename token: `re.sub(r'[^a-zA-Z0-9]+', '_', s).strip('_')`
- `_build_tree(combos)` recursively nests the combo list; the leaf level becomes a plain array
- Each combo file contains **all years** for that combo — lazy-loaded only when the user selects that combo
- Loaded as `<script>` tags — no server or bundler needed, works from `file://`

---

## Module Data Structures

**General rules:**

- **Map files (split pattern)**: each combo `(dim1, …, dimN)` is a separate JS file containing `{ year: {img_str, bounds, intOrFloat, legend, min_max}, … }`. An index file lists all valid combos.
- **Chart files**: end at `[series array]` where each series is `{name, data, type, color}`.
- **MapService**: every non-mask entry is `{ indexPath, indexName, layerPrefix }`. Exception: `mask` entries in Biodiversity GBF2 remain `{ path, name }` (GeoJSON overlay, not a split map tile).

### Area

| | Ag | Am | NonAg |
|-|----|----|-------|
| **Chart** | `Region → Water → [series]` | `Region → Water → LU → [series]` (indexed by am) | `Region → [series]` |
| **Map** | `Water → LU → Year` | `AgMgt → Water → LU → Year` | `LU → Year` |

### Economics

**UI selection order**: Category → Map Type → (AgMgt) → Water → (Source) → Landuse

The economics module is unique: map files are split by both Category and Map Type; charts always show all series regardless of map type.

**Chart files** (one per category, contain both cost & revenue mixed):

| File | Hierarchy |
|------|-----------|
| `Economics_Ag` | `Region → "ALL" → "ALL" → [series]` |
| `Economics_Am` | `Region → "ALL" → "ALL" → [series]` |
| `Economics_Non_Ag` | `Region → [series]` |

**Map files** — Ag (5 map types):

| Map Type | Hierarchy | Source level? |
|----------|-----------|---------------|
| `map_economics_Ag_profit` | `Water → LU → Year` | No |
| `map_economics_Ag_revenue` | `Water → Source → LU → Year` | Yes |
| `map_economics_Ag_cost` | `Water → Source → LU → Year` | Yes |
| `map_economics_Ag_transition_ag2ag` | `Water → Source → LU → Year` | Yes |
| `map_economics_Ag_transition_ag2nonag` | `Water → Source → LU → Year` | Yes (`from_lu` treated as LU) |

**Map files** — Am (3 map types, no Source level):

`map_economics_Am_{profit/revenue/cost}` → `AgMgt → Water → LU → Year`

**Map files** — NonAg (3 map types, no Water/Source level):

`map_economics_NonAg_{profit/revenue/cost}` → `LU → Year`

**MapService structure**: `Economics → {Ag/Am/NonAg} → {MapType} → {path, name}`
- `availableMapTypes = Object.keys(mapRegister[category])` — dynamic (5 for Ag, 3 for Am/NonAg)
- Source level shown only when `category === "Ag" && mapType !== "Profit"` (`hasSourceLevel` computed)

### GHG

| | Ag | Am | NonAg |
|-|----|----|-------|
| **Chart** | `Region → Water → Source → [series(name=LU)]` | `Region → Water → LU → [series(name=AgMgt)]` | `Region → [series(name=LU)]` |
| **Map** | `Water → Source → LU → Year` | `AgMgt → Water → LU → Year` | `LU → Year` |

Source values (Ag only): emission type e.g. `"Enteric Fermentation"`, `"Manure"`, `"Chemical Application"`.

**GHG Ag** adds `selectSource` state (between Water and LU). Cascade: Category → Water → Source → LU. `previousSelections["Ag"] = { water, source, landuse }`.

**GHG Am** has no Source level — cascade: Category → AgMgt → Water → LU.

### Production

| | Ag | Am | NonAg |
|-|----|----|-------|
| **Chart** | `Region → Water → [series]` | `Region → Water → LU → [series]` | `Region → [series]` |
| **Map** | `Water → LU → Commodity → Year` | `AgMgt → Water → LU → Commodity → Year` | `LU → Commodity → Year` |

### Water

| | Ag | Am | NonAg |
|-|----|----|-------|
| **Chart** | `Region → Water → [series(name=LU)]` | `Region → Water → LU → [series(name=AgMgt)]` | `Region → [series(name=LU)]` |
| **Map** | `Water → LU → Year` | `AgMgt → Water → LU → Year` | `LU → Year` |

**Note**: Water Am chart series are indexed by **AgMgt name** (filtered by `selectAgMgt`), unlike other modules where Am series are indexed by LU.

### Biodiversity (Metric + Conditional Loading)

Biodiversity adds a **Metric** selection level on top of the standard Category → AgMgt → Water → LU hierarchy. Metrics map to GBF targets; `quality` is always available.

**MapService/ChartService access**: `mapRegister["Biodiversity"][metric][category]` (not `["Biodiversity"]["quality"]`).

**Available metrics and their `settings.py` gating key:**

| Metric key | Display label | Settings key | Always loaded? |
|-----------|--------------|-------------|---------------|
| `quality` | Quality | — | Yes |
| `GBF2` | GBF2 | `BIODIVERSITY_TARGET_GBF_2` | If ≠ 'off' |
| `GBF3_NVIS` | GBF3 NVIS | `BIODIVERSITY_TARGET_GBF_3_NVIS` | If ≠ 'off' |
| `GBF3_IBRA` | GBF3 IBRA | `BIODIVERSITY_TARGET_GBF_3_IBRA` | If ≠ 'off' |
| `GBF4_SNES` | GBF4 SNES | `BIODIVERSITY_TARGET_GBF_4_SNES` | If ≠ 'off' |
| `GBF4_ECNES` | GBF4 ECNES | `BIODIVERSITY_TARGET_GBF_4_ECNES` | If ≠ 'off' |
| `GBF8_GROUP` | GBF8 Group | `BIODIVERSITY_TARGET_GBF_8` | If ≠ 'off' |
| `GBF8_SPECIES` | GBF8 Species | `BIODIVERSITY_TARGET_GBF_8` | If ≠ 'off' |

**Per-metric available categories**: GBF3/GBF4 metrics have no `Sum` mosaic layer.
`availableCategories` is a computed that filters `["Sum","Ag","Ag Mgt","Non-Ag"]` against
the metric's `mapRegister` keys, so the `Sum` button is hidden for those metrics.

**Species dimension** (`METRICS_WITH_SPECIES = ['GBF3_NVIS','GBF4_SNES','GBF4_ECNES','GBF8_GROUP','GBF8_SPECIES']`):
these metrics carry an extra dimension between `region`/`water`/`agMgt` and `landuse` — the
species / vegetation group / community / functional group. The label adapts per metric
(`'Veg group:'` for GBF3 NVIS, `'Species:'` for GBF4 SNES / GBF8 SPECIES, `'Community:'`
for GBF4 ECNES, `'Group:'` for GBF8 GROUP). The selector is rendered as a **floating
bottom-right scrollable panel** (~280×260px, max-h with `overflow-y-auto`, text-left
buttons) that slides left when the drawer opens.

**Selection state**: `selectMetric` → `selectCategory` → `selectAgMgt` → `selectWater` → `selectSpecies` (when applicable) → `selectLanduse`.

**Cascade on metric change**: calls `doCascade(selectCategory.value)` to re-populate available options from `mapRegister[selectMetric][category]`. When the new metric has a species dim, `availableSpecies` is populated from the next level of the map data after the water/agMgt path; on metric switch the previous species is preserved if still valid, otherwise reset to the first option.

**Map hierarchy** (mirrors physical `xr_biodiversity_*.nc` files):
- Ag without species:    `Water → LU → Year`
- Ag with species:       `Water → Species → LU → Year`
- Am without species:    `AgMgt → Water → LU → Year`
- Am with species:       `AgMgt → Water → Species → LU → Year`
- NonAg without species: `LU → Year`
- NonAg with species:    `Species → LU → Year`

**Chart hierarchy** (mirrors map; `lu` is aggregated as a stacked column series — `plotOptions.column.stacking = 'normal'`):
- Ag without species:    `Region → Water → [series(name=LU)]`
- Ag with species:       `Region → Species → Water → [series(name=LU)]`
- Am without species:    `Region → AgMgt → Water → [series(name=LU)]`
- Am with species:       `Region → Species → AgMgt → Water → [series(name=LU)]`
- NonAg without species: `Region → [series(name=LU)]`
- NonAg with species:    `Region → Species → [series(name=LU)]`

In `selectChartData`, when `hasSpecies.value`, the lookup descends into the species level
right after `region` before applying the standard water / agMgt / landuse filters.

**`selectMapData` access:**
```javascript
// Ag (no species):    mapData?.[selectWater]?.[selectLanduse]?.[year]
// Ag (with species):  mapData?.[selectWater]?.[selectSpecies]?.[selectLanduse]?.[year]
// Am (no species):    mapData?.[selectAgMgt]?.[selectWater]?.[selectLanduse]?.[year]
// Am (with species):  mapData?.[selectAgMgt]?.[selectWater]?.[selectSpecies]?.[selectLanduse]?.[year]
// NonAg (no species): mapData?.[selectLanduse]?.[year]
// NonAg (w/ species): mapData?.[selectSpecies]?.[selectLanduse]?.[year]
// (mapData = window[mapRegister[selectMetric][selectCategory]["name"]])
```

**Greyscale ramp for unselected NRMs**: when a metric is restricted to a subset of NRMs
(GBF3 NVIS, GBF4 SNES, GBF4 ECNES in `'NRM'` mode), the write-side attaches an
`is_selected` cell coord and the renderer maps unselected non-zero cells through the grey
palette segment (codes 151-200). See [CLAUDE_OUTPUT.md](CLAUDE_OUTPUT.md) §
"Greyscale Ramp for Unselected Cells (`is_selected` Coord)" for the full pipeline.

### DVAR (Map-only)

The view composes a combined data structure from multiple files:

| Source file | Keys available |
|-------------|----------------|
| `map_dvar_lumap` | `"Land-use"`, `"Water-supply"`, `"Agricultural Land-use"`, `"Agricultural Management"`, `"Non-Agricultural Land-use"` → `Year` |
| `map_dvar_Ag` | `LU → Year` |
| `map_dvar_Am` | `AgMgt → Year` |
| `map_dvar_NonAg` | `LU → Year` |

Final composed hierarchy: `Category → LU/AgMgt → Year`

---

## Progressive Selection Pattern

All views use the same pattern: cascading reactive selections drive `selectMapData` (computed) which reads the nested JS object.

### Hierarchy types

| Pattern | Modules | Button order |
|---------|---------|--------------|
| Standard Full | Area, Production, Water | Category → AgMgt → Water → LU |
| GHG | GHG | Category → AgMgt → Water → LU (Am); Category → Water → Source → LU (Ag) |
| Biodiversity | Biodiversity | Metric → Category → AgMgt → Water → (Species) → LU |
| NonAg Simplified | NonAg in all modules | Category → LU |
| DVAR | DVAR | Category → LU/AgMgt → Year |
| Economics Extended | Economics | Category → MapType → (AgMgt) → Water → (Source) → LU |

### Standard cascade watchers (Area, Production, Water)

All views use three helpers provided by the split-file pattern:

```javascript
// helpers.js factory — creates per-view lazy loader
const { currentLayerData, ensureComboLayer } = window.createMapLayerLoader(VIEW_NAME);
const selectMapData = computed(() => currentLayerData.value?.[selectYear.value] ?? {});

// getTree(cat) — reads the dim tree from the already-loaded index
function getTree(cat) {
  return window[mapRegister[cat]?.indexName]?.tree ?? (cat === "Non-Ag" ? [] : {});
}

// ensureIndexLoaded(cat) — lazy-loads the __index.js for a category on first use
async function ensureIndexLoaded(cat) {
  const entry = mapRegister[cat];
  if (entry && !window[entry.indexName]) {
    isLoadingData.value = true;
    await loadScript(entry.indexPath, entry.indexName, VIEW_NAME);
    isLoadingData.value = false;
  }
}
```

Four watchers in fixed order — each handles all its downstream options:

```javascript
// 1. Category → lazy-loads index, reads tree, populates AgMgt/Water/LU, triggers combo load
watch(selectCategory, async (newCat, oldCat) => {
  if (oldCat) previousSelections.value[oldCat] = { agMgt: ..., water: ..., landuse: ... };
  await ensureIndexLoaded(newCat);
  const tree = getTree(newCat);
  if (newCat === "Ag Mgt") {
    availableAgMgt.value   = Object.keys(tree);
    selectAgMgt.value      = restore(prev.agMgt, availableAgMgt) || availableAgMgt.value[0];
    availableWater.value   = Object.keys(tree[selectAgMgt.value] || {});
    selectWater.value      = restore(prev.water, availableWater) || availableWater.value[0];
    availableLanduse.value = tree[selectAgMgt.value]?.[selectWater.value] || [];
    selectLanduse.value    = restore(prev.landuse, availableLanduse) || availableLanduse.value[0];
    await ensureComboLayer(mapRegister["Ag Mgt"].layerPrefix, [selectAgMgt.value, selectWater.value, selectLanduse.value]);
  } else if (newCat === "Ag") { /* similar: water → lu → ensureComboLayer([water, lu]) */ }
  else if (newCat === "Non-Ag") { /* lu only → ensureComboLayer([lu]) */ }
});

// 2. AgMgt → Water → LU → ensureComboLayer  (Ag Mgt only)
watch(selectAgMgt, async (newAgMgt) => { ... });

// 3. Water → LU → ensureComboLayer
watch(selectWater, async (newWater) => { ... });

// 4. LU → ensureComboLayer + save
watch(selectLanduse, async (newLanduse) => {
  await ensureComboLayer(mapRegister[cat].layerPrefix, [/* dims for current cat */]);
});
```

**Rule**: Never manually clear `available*` arrays — always overwrite with new values. Each watcher handles ALL its downstream selections in one pass, ending with an `ensureComboLayer` call.

### GHG cascade

GHG follows the standard pattern but **Ag adds `selectSource`** between Water and LU. The tree shape differs by category:

- **Ag tree**: `{ water: { source: [lu] } }` — three-level
- **Am tree**: `{ am: { lm: [lu] } }` — same as standard Am

```javascript
// Ag category cascade: Water → Source → LU → ensureComboLayer
watch(selectWater, async (newWater) => {
  if (selectCategory.value === "Ag") {
    const tree = getTree("Ag");
    availableSource.value = Object.keys(tree[newWater] || {});
    selectSource.value = restore(prev.source, availableSource) || availableSource.value[0];
    availableLanduse.value = tree[newWater]?.[selectSource.value] || [];
    selectLanduse.value = restore(prev.landuse) || availableLanduse.value[0];
    await ensureComboLayer(mapRegister["Ag"].layerPrefix, [newWater, selectSource.value, selectLanduse.value]);
  } // Am: standard Water → LU (no source)
});

watch(selectSource, async (newSource) => {
  if (selectCategory.value !== "Ag") return;
  const tree = getTree("Ag");
  availableLanduse.value = tree[selectWater.value]?.[newSource] || [];
  selectLanduse.value = restore(prev.landuse) || availableLanduse.value[0];
  await ensureComboLayer(mapRegister["Ag"].layerPrefix, [selectWater.value, newSource, selectLanduse.value]);
});
```

### Biodiversity cascade

Biodiversity wraps all cascade logic in a `doCascade(category)` helper to support the extra `selectMetric` dimension. It uses the same `getTree` / `ensureComboLayer` pattern but keyed on metric:

```javascript
function getTree(metric, cat) {
  return window[mapRegister[metric]?.[cat]?.indexName]?.tree ?? (cat === "Non-Ag" ? [] : {});
}

async function ensureIndexLoaded(metric, cat) {
  const entry = mapRegister[metric]?.[cat];
  if (entry && !window[entry.indexName]) {
    isLoadingData.value = true;
    await loadScript(entry.indexPath, entry.indexName, VIEW_NAME);
    isLoadingData.value = false;
  }
}

async function doCascade(cat) {
  const metric = selectMetric.value;
  await ensureIndexLoaded(metric, cat);
  const tree = getTree(metric, cat);
  // ... build combo arrays (including optional species dim), then:
  await ensureComboLayer(mapRegister[metric][cat].layerPrefix, buildCombo(cat, tree, hasSpecies.value));
}

// Both category and metric changes call doCascade
watch(selectCategory, async (newCat, oldCat) => { /* save prev */ await doCascade(newCat); });
watch(selectMetric,   async () => await doCascade(selectCategory.value));
```

`buildCombo(cat, tree, withSpecies)` returns the ordered dim array for `ensureComboLayer`:

- **Sum**: `[species, lu]` (with species) or `[lu]`
- **Ag**: `[water, species, lu]` (with species) or `[water, lu]`
- **Ag Mgt**: `[agMgt, water, species, lu]` (with species) or `[agMgt, water, lu]`
- **Non-Ag**: `[species, lu]` (with species) or `[lu]`

### Economics cascade watchers

Economics adds a `selectMapType` level and a conditional `selectSource` level. Each unique `(category, mapType)` pair maps to a distinct `MapService` entry with its own `layerPrefix`.

```javascript
// Combined watcher: category + mapType drive everything downstream
watch([selectCategory, selectMapType], async ([newCat, newMapType], [oldCat]) => {
  if (oldCat && oldCat !== newCat) saveSelections(oldCat);

  availableMapTypes.value = Object.keys(mapRegister[newCat] || {});

  // If mapType invalid for new category → restore previous or use first, then re-fire
  if (!availableMapTypes.value.includes(newMapType)) {
    selectMapType.value = previousSelections.value[newCat]?.mapType || availableMapTypes.value[0];
    return;
  }
  // lazy-load index for (cat, mapType), read tree, populate selections
  await ensureIndexLoaded(newCat, newMapType);
  const tree = getTree(newCat, newMapType);
  cascadeAll(newCat, newMapType, tree);
  // end with ensureComboLayer for the resolved combo
}, { immediate: true });

// Source → LU → ensureComboLayer  (Ag non-Profit only)
const hasSourceLevel = computed(() => selectCategory.value === "Ag" && selectMapType.value !== "Profit");
watch(selectSource, async (newSource) => {
  if (!hasSourceLevel.value) return;
  const tree = getTree(selectCategory.value, selectMapType.value);
  availableLanduse.value = tree[selectWater.value]?.[newSource] || [];
  selectLanduse.value = restore(prev.landuse) || availableLanduse.value[0];
  await ensureComboLayer(mapRegister[selectCategory.value][selectMapType.value].layerPrefix,
    [selectWater.value, newSource, selectLanduse.value]);
});
```

**Cascading flow**:
`[category, mapType]` → AgMgt → Water → Source (if Ag non-Profit) → LU → `ensureComboLayer`

**`previousSelections` fields**:

- `"Ag"`: `{ mapType, water, source, landuse }`
- `"Ag Mgt"`: `{ mapType, agMgt, water, landuse }`
- `"Non-Ag"`: `{ mapType, landuse }`

### `selectMapData` access pattern

With the split-file pattern, **all views use the same single expression**:

```javascript
const selectMapData = computed(() => currentLayerData.value?.[selectYear.value] ?? {});
```

`currentLayerData` is a `ref` set by `ensureComboLayer` to the loaded combo's year-keyed object.
The combo dimensions are passed explicitly when calling `ensureComboLayer`:

```javascript
// Area / Production / Water / GHG / Biodiversity — Ag
await ensureComboLayer(layerPrefix, [selectWater.value, selectLanduse.value]);
// Area / Production / Water / GHG / Biodiversity — Am
await ensureComboLayer(layerPrefix, [selectAgMgt.value, selectWater.value, selectLanduse.value]);
// Area / Production / Water / GHG / Biodiversity — NonAg
await ensureComboLayer(layerPrefix, [selectLanduse.value]);

// GHG Ag (has Source level between Water and LU)
await ensureComboLayer(layerPrefix, [selectWater.value, selectSource.value, selectLanduse.value]);

// Biodiversity with species (GBF3_NVIS, GBF4_SNES, GBF4_ECNES, GBF8) — Ag
await ensureComboLayer(layerPrefix, [selectWater.value, selectSpecies.value, selectLanduse.value]);
// Biodiversity with species — Am
await ensureComboLayer(layerPrefix, [selectAgMgt.value, selectWater.value, selectSpecies.value, selectLanduse.value]);

// Economics Ag Profit
await ensureComboLayer(layerPrefix, [selectWater.value, selectLanduse.value]);
// Economics Ag Revenue/Cost/Transition (has Source)
await ensureComboLayer(layerPrefix, [selectWater.value, selectSource.value, selectLanduse.value]);
// Economics Am
await ensureComboLayer(layerPrefix, [selectAgMgt.value, selectWater.value, selectLanduse.value]);
// Economics NonAg
await ensureComboLayer(layerPrefix, [selectLanduse.value]);
```

The loaded file name is derived as: `<layerPrefix>__<safe(dim1)>__…__<safe(dimN)>.js`
where `safe(s) = s.replace(/[^a-zA-Z0-9]+/g, '_').trim('_')`.
