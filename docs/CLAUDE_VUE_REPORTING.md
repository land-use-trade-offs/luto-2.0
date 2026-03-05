# LUTO2 Vue.js Reporting System

Vue.js 3 dashboard (`DATA_REPORT/REPORT_HTML/index.html`) that renders LUTO simulation outputs as interactive maps and charts. No build step — all JS files are loaded via `<script>` tags from local `data/` directories.

## File Structure

| Path | Purpose |
|------|---------|
| `VUE_modules/views/` | View components (one per module) |
| `VUE_modules/services/MapService.js` | Map file registry: `module → category → mapType → {path, name}` |
| `VUE_modules/services/ChartService.js` | Chart file registry: `module → category → {path, name}` |
| `VUE_modules/data/map_layers/` | Pre-rendered base64 PNG map tiles (`.js` files) |
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

### Step 8 — Write `.js` file

```python
output = tuple_dict_to_nested(output)   # flat tuple keys → nested dict
# writes: window["map_economics_Ag_cost"] = { "ALL": { "Area cost": { "Apples": { "2020": {...} } } } };
```

Loaded as a `<script>` tag — no server or bundler needed, works from `file://`.

---

## Module Data Structures

**General rules:**
- **Map files**: end at `Year → {img_str, bounds, intOrFloat, legend, min_max}`. Max hierarchy: `am → lm → [source] → lu → year` (`source` only in Ag for GHG/Economics; absent from Am).
- **Chart files**: end at `[series array]` where each series is `{name, data, type, color}`.
- **MapService**: `module → Category → MapType → {path, name}` (Economics); simpler `module → Category → {path, name}` for other modules.

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

**Selection state**: `selectMetric` → `selectCategory` → `selectAgMgt` → `selectWater` → `selectLanduse`.

**Cascade on metric change**: calls `doCascade(selectCategory.value)` to re-populate available options from `mapRegister[selectMetric][category]`.

**Chart hierarchy:**
- Ag: `Region → Water → [series(name=LU)]`
- Am: `Region → AgMgt → Water → [series(name=LU)]`
- NonAg: `Region → [series(name=LU)]`

**Map files**: standard patterns — `Water → LU → Year` for Ag, `AgMgt → Water → LU → Year` for Am, `LU → Year` for NonAg.

**`selectMapData` access:**
```javascript
// Ag:    mapData?.[selectWater]?.[selectLanduse]?.[year]
// Am:    mapData?.[selectAgMgt]?.[selectWater]?.[selectLanduse]?.[year]
// NonAg: mapData?.[selectLanduse]?.[year]
// (mapData = window[mapRegister[selectMetric][selectCategory]["name"]])
```

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
| Biodiversity | Biodiversity | Metric → Category → AgMgt → Water → LU |
| NonAg Simplified | NonAg in all modules | Category → LU |
| DVAR | DVAR | Category → LU/AgMgt → Year |
| Economics Extended | Economics | Category → MapType → (AgMgt) → Water → (Source) → LU |

### Standard cascade watchers (Area, Production, Water)

Four watchers in fixed order — each handles all its downstream options:

```javascript
// 1. Category → saves previous, populates AgMgt/Water/LU
watch(selectCategory, (newCat, oldCat) => {
  if (oldCat) previousSelections.value[oldCat] = { agMgt: ..., water: ..., landuse: ... };
  if (newCat === "Ag Mgt") {
    availableAgMgt.value   = Object.keys(window[mapRegister["Ag Mgt"]["name"]] || {});
    selectAgMgt.value      = restore(prev.agMgt, availableAgMgt) || availableAgMgt.value[0];
    availableWater.value   = Object.keys(amData?.[selectAgMgt.value] || {});
    selectWater.value      = restore(prev.water, availableWater) || availableWater.value[0];
    availableLanduse.value = Object.keys(amData?.[selectAgMgt.value]?.[selectWater.value] || {});
    selectLanduse.value    = restore(prev.landuse, availableLanduse) || availableLanduse.value[0];
  } else if (newCat === "Ag") { /* water → lu */ }
  else if (newCat === "Non-Ag") { /* lu only */ }
});

// 2. AgMgt → Water → LU  (Ag Mgt only)
watch(selectAgMgt, ...);

// 3. Water → LU
watch(selectWater, ...);

// 4. LU → save only
watch(selectLanduse, ...);
```

**Rule**: Never manually clear `available*` arrays — always overwrite with new values. Each watcher handles ALL its downstream selections in one pass.

### GHG cascade

GHG follows the standard pattern but **Ag adds `selectSource`** between Water and LU:

```javascript
// Ag category cascade: Water → Source → LU
watch(selectWater, (newWater) => {
  if (selectCategory.value === "Ag") {
    availableSource.value = Object.keys(agData?.[newWater] || {});
    selectSource.value = restore(prev.source, availableSource) || availableSource.value[0];
    availableLanduse.value = Object.keys(agData?.[newWater]?.[selectSource.value] || {});
    selectLanduse.value = restore(prev.landuse) || availableLanduse.value[0];
  } // Am: standard Water → LU (no source)
});

watch(selectSource, (newSource) => {
  if (selectCategory.value !== "Ag") return;
  availableLanduse.value = Object.keys(agData?.[selectWater.value]?.[newSource] || {});
  selectLanduse.value = restore(prev.landuse) || availableLanduse.value[0];
});
```

### Biodiversity cascade

Biodiversity wraps all cascade logic in a `doCascade(category)` helper to support the extra `selectMetric` dimension:

```javascript
function doCascade(category) {
  const mr = mapRegister[selectMetric.value];  // metric-level lookup
  const agData   = window[mr?.["Ag"]?.["name"]];
  const amData   = window[mr?.["Ag Mgt"]?.["name"]];
  const nonAgData = window[mr?.["Non-Ag"]?.["name"]];
  // ... same cascade logic as standard, but using metric-keyed data
}

// Both category and metric changes call doCascade
watch(selectCategory, (newCat, oldCat) => { /* save prev */ doCascade(newCat); });
watch(selectMetric,   () => doCascade(selectCategory.value));
```

### Economics cascade watchers

Economics adds a `selectMapType` level and a conditional `selectSource` level:

```javascript
// Combined watcher: category + mapType drive everything downstream
watch([selectCategory, selectMapType], ([newCat, newMapType], [oldCat]) => {
  if (oldCat && oldCat !== newCat) saveSelections(oldCat);

  availableMapTypes.value = Object.keys(mapRegister[newCat] || {});

  // If mapType invalid for new category → restore previous or use first, then re-fire
  if (!availableMapTypes.value.includes(newMapType)) {
    selectMapType.value = previousSelections.value[newCat]?.mapType
      || availableMapTypes.value[0];
    return;
  }
  cascadeAll(newCat, newMapType);
}, { immediate: true });

// Source → LU  (Ag non-Profit only)
const hasSourceLevel = computed(() => selectCategory.value === "Ag" && selectMapType.value !== "Profit");
watch(selectSource, (newSource) => {
  if (!hasSourceLevel.value) return;
  availableLanduse.value = Object.keys(mapData[selectWater.value][newSource] || {});
  selectLanduse.value = restore(prev.landuse) || availableLanduse.value[0];
});
```

**Cascading flow**:
`[category, mapType]` → AgMgt → Water → Source (if Ag non-Profit) → LU → save

**`previousSelections` fields**:
- `"Ag"`: `{ mapType, water, source, landuse }`
- `"Ag Mgt"`: `{ mapType, agMgt, water, landuse }`
- `"Non-Ag"`: `{ mapType, landuse }`

### `selectMapData` access patterns

Always use optional chaining: `mapData?.[a]?.[b]?.[c] || {}`.

```javascript
// Standard Ag (Area, Production, Water, Biodiversity)
mapData?.[selectWater]?.[selectLanduse]?.[year]
// Standard Am
mapData?.[selectAgMgt]?.[selectWater]?.[selectLanduse]?.[year]
// Standard NonAg
mapData?.[selectLanduse]?.[year]

// GHG Ag (has Source level between Water and LU)
mapData?.[selectWater]?.[selectSource]?.[selectLanduse]?.[year]
// GHG Am (no Source)
mapData?.[selectAgMgt]?.[selectWater]?.[selectLanduse]?.[year]

// Biodiversity: mapData = window[mapRegister[selectMetric][selectCategory]["name"]]
//   Ag:    mapData?.[selectWater]?.[selectLanduse]?.[year]
//   Am:    mapData?.[selectAgMgt]?.[selectWater]?.[selectLanduse]?.[year]
//   NonAg: mapData?.[selectLanduse]?.[year]

// Economics Ag Profit
mapData?.[selectWater]?.[selectLanduse]?.[year]
// Economics Ag Revenue/Cost/Transition (has Source level)
mapData?.[selectWater]?.[selectSource]?.[selectLanduse]?.[year]
// Economics Am
mapData?.[selectAgMgt]?.[selectWater]?.[selectLanduse]?.[year]
// Economics NonAg
mapData?.[selectLanduse]?.[year]
```
