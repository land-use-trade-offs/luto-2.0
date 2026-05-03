# LUTO 2.0 Reporting Dashboard

## Overview

This directory contains the web-based reporting dashboard for the Land Use Trade-Offs (LUTO) 2.0 model. It is generated automatically as part of `DATA_REPORT/` when a LUTO simulation run completes. The dashboard provides interactive charts, maps, and data exploration tools for understanding the environmental, economic, and social impacts of different land use scenarios in Australia.

Open the report by double-clicking `index.html` — no server or internet connection required.

## Purpose

The dashboard visualises LUTO 2.0 model outputs across nine analysis modules:
- **Area**: Land use area distribution and temporal changes
- **Economics**: Revenue, cost, and profit across agricultural and non-agricultural sectors
- **GHG**: Greenhouse gas emissions by land use and management type
- **Water**: Water yield changes by NRM region
- **Production**: Agricultural commodity output and demand achievement
- **Biodiversity**: GBF2/3/4/8 biodiversity framework target indicators and habitat quality scores
- **Renewable Energy**: Solar PV and wind MWh generation, including pre-existing installations
- **Transitions**: From→To land use transition area with dual heatmap + spatial map
- **Map (DVAR)**: Spatial decision-variable allocation (categorical land use maps)

All views share a progressive selection pattern — users drill from national overviews into specific regions, categories, management types, and land uses.

## Technology Stack

| Library | Version | Role |
|---------|---------|------|
| Vue.js  | 3.5.18  | Frontend framework (Composition API, Options-style `window.XxxView`) |
| Vue Router | 4.5.1 | Hash-based client-side routing |
| Tailwind CSS | 3.4.16 | Utility-first styling |
| Element Plus | 2.10.4 | UI component library |
| Highcharts | 12.3.0 | Time-series charts and heatmaps |
| Leaflet | 1.9.4 | Interactive region map |

All libraries are locally bundled in `lib/` — no CDN or build step required.

## Project Structure

```
VUE_modules/
├── index.html                          # Entry point — loads libs, registers components, mounts app
├── index.js                            # App bootstrap: provides globalSelectedRegion / globalMapViewpoint
│
├── components/                         # Globally registered Vue components
│   ├── chart_container.js              # <chart-container>   — Highcharts time-series wrapper
│   ├── heatmap_container.js            # <heatmap-container> — Highcharts heatmap (From→To matrix)
│   ├── regions_map.js                  # <regions-map>       — NRM region selector with base64 overlay
│   ├── map_geojson.js                  # <map-geojson>       — Leaflet NRM choropleth
│   ├── sidebar.js                      # <side-bar>          — Navigation sidebar
│   ├── ranking_cards.js                # <ranking-cards>     — Top-N land-use ranking cards
│   ├── filterable_dropdown.js          # <filterable-dropdown> — Searchable dropdown
│   └── helpers.js                      # loadScriptWithTracking() + createMapLayerLoader() — on-demand JS loaders
│
├── views/                              # Route-level page components
│   ├── Home.js                         # / — overview charts, transition heatmap, ranking cards, run settings
│   ├── Area.js                         # /area
│   ├── Economics.js                    # /economics
│   ├── GHG.js                          # /ghg
│   ├── Water.js                        # /water
│   ├── Production.js                   # /production
│   ├── Biodiversity.js                 # /biodiversity
│   ├── Renewable.js                    # /renewable
│   ├── Transition.js                   # /transition
│   ├── Map.js                          # /map  (Decision Variables)
│   ├── Settings.js                     # /settings
│   └── NotFound.js                     # /* (404)
│
├── services/
│   ├── ChartService.js                 # chartCategories registry — path + window-name per dataset
│   ├── MapService.js                   # mapCategories registry  — path + window-name per map layer
│   └── MemoryService.js                # cleanupViewData(VIEW_NAME) — removes window globals on unmount
│
├── routes/
│   └── route.js                        # createRouter(createWebHashHistory(), routes)
│
├── data/                               # Generated data files (produced by create_report_*.py)
│   ├── Supporting_info.js              # Model run settings, year list, scenario metadata
│   ├── chart_option/
│   │   ├── Chart_default_options.js
│   │   └── chartMemLogOptions.js
│   ├── geo/
│   │   ├── NRM_AUS.js                  # GeoJSON — Australian NRM boundaries
│   │   └── biodiversity_GBF2_mask.js   # GBF2 priority-area mask polygon
│   ├── map_layers/                     # Base64 spatial raster tiles — split-file pattern
│   │   │                               #   <prefix>__index.js          — dim tree + valid combos
│   │   │                               #   <prefix>__<d1>__<d2>….js    — all years for that combo
│   │   ├── map_area_Ag__index.js  +  map_area_Ag__<lm>__<lu>.js  (×N combos)
│   │   ├── map_area_Am__index.js  +  map_area_Am__<am>__<lm>__<lu>.js
│   │   ├── map_area_NonAg__index.js  +  map_area_NonAg__<lu>.js
│   │   ├── map_GHG_*/map_water_yield_*/map_quantities_* — same split pattern
│   │   ├── map_economics_Ag_profit/revenue/cost/transition_* — split per combo
│   │   ├── map_economics_Am_profit/revenue/cost — split per combo
│   │   ├── map_economics_NonAg_profit/revenue/cost — split per combo
│   │   ├── map_bio_GBF2_Sum/Ag/Am/NonAg — split per combo
│   │   ├── map_bio_GBF3_NVIS/GBF4_SNES/GBF4_ECNES/GBF8_* — split per combo
│   │   ├── map_bio_overall_* — split per combo
│   │   ├── map_dvar_Ag/Am/NonAg/lumap — split per combo
│   │   ├── map_renewable_energy_Am__index.js + per-combo files
│   │   ├── map_transition_area_ag2ag__index.js + per-combo files
│   │   └── biodiversity_GBF2_mask.js   # GeoJSON overlay — NOT split (mask entry)
│   ├── Area_Ag/Am/NonAg.js             # Chart: region → lm → lu → [series]
│   ├── Area_overview_*.js
│   ├── Area_ranking.js
│   ├── Economics_Ag_profit/revenue/cost.js
│   ├── Economics_Ag_transition_ag2ag/nonag2ag.js
│   ├── Economics_Am_profit/revenue/cost.js
│   ├── Economics_NonAg_profit/revenue/cost.js
│   ├── Economics_NonAg_transition_ag2nonag/nonag2nonag.js
│   ├── Economics_Sum.js
│   ├── Economics_overview_*.js / Economics_ranking.js
│   ├── GHG_Sum/Ag/Am/NonAg.js
│   ├── GHG_overview_*.js / GHG_ranking.js
│   ├── Production_Sum/Ag/Am/NonAg.js
│   ├── Production_overview_*.js / Production_ranking.js
│   ├── Water_Sum/Ag/Am/NonAg_NRM.js
│   ├── Water_overview_NRM_*.js / Water_ranking_NRM.js
│   ├── Water_overview_watershed.js
│   ├── BIO_GBF2/GBF3_NVIS/GBF3_IBRA/GBF4_SNES/GBF4_ECNES/GBF8_GROUP/GBF8_SPECIES_Ag/Am/NonAg.js
│   ├── BIO_*_overview_*.js / BIO_ranking.js
│   ├── Renewable_energy_Am.js
│   ├── Transition_ag2ag_area.js        # Chart heatmap: region → from_water → to_water → year → leaf
│   └── Transition_start_end_area.js    # Chart heatmap: region → from_water → to_water → leaf (no year)
│
├── lib/                                # Locally bundled libraries (offline-capable)
├── assets/                             # Raw JSON (input to dataTransform scripts)
├── dataTransform/                      # JSON → JS conversion utilities
└── resources/                          # Logo, fonts, icons
```

## Views and Routes

| Route | Component | Description |
|-------|-----------|-------------|
| `/` | `HomeView` | Overview charts for each module, transition heatmap, ranking cards, memory profile, run settings |
| `/area` | `AreaView` | Land use area by Ag / Ag Mgt / Non-Ag |
| `/economics` | `EconomicsView` | Profit / revenue / cost + transition cost by Sum / Ag / Ag Mgt / Non-Ag |
| `/ghg` | `GHGView` | GHG emissions by Sum / Ag / Ag Mgt / Non-Ag |
| `/water` | `WaterView` | Water yield by NRM region (NRM tab) + watershed overview tab |
| `/production` | `ProductionView` | Commodity production + demand achievement by Sum / Ag / Ag Mgt / Non-Ag |
| `/biodiversity` | `BiodiversityView` | Biodiversity quality + GBF2/3/4/8 targets; Metric selection gates data load |
| `/renewable` | `RenewableView` | Renewable energy MWh (solar + wind); Am-only hierarchy |
| `/transition` | `TransitionView` | From→To transition heatmap + spatial map; SubCats auto-discovered from MapService |
| `/map` | `MapView` | Categorical decision-variable spatial maps (Ag / Ag Mgt / Non-Ag / Land-use) |
| `/settings` | `SettingsView` | Model run parameter browser |

### Progressive Selection Hierarchy per View

| View | Selection levels |
|------|-----------------|
| Area | Category → Water → Landuse |
| Economics | Category → MapType → (AgMgt) → Water → (Source) → Landuse |
| GHG | Category → (AgMgt) → Water → Landuse |
| Water | Category → (AgMgt) → Water → Landuse |
| Production | Category → (AgMgt) → Water → Landuse |
| Biodiversity | **Metric** → Category → (AgMgt) → Water → Landuse |
| Renewable | AgMgt → Water → Landuse (Am-only) |
| Transition | SubCat → FromWater → ToWater + heatmap cell → Year |
| Map (DVAR) | Category → (AgMgt) → Water → Landuse |

`(AgMgt)` / `(Source)` appear only for the categories that need them. All views inject `globalSelectedRegion` for the NRM region selector.

### Home View Details
- Displays overview charts for Area, Economics, GHG, Water, Biodiversity, Production
- Shows a live transition heatmap (Ag2Ag start→end area) at the bottom
- Ranking cards show top land uses per module sub-category
- Memory profile chart shows model run RAM usage over time
- Searchable model run parameters panel (from `Supporting_info.js`)

## Service Layer

### MapService (`services/MapService.js`)
Registry of all spatial-layer JS files, keyed by `mapCategories[module][category][subcategory]`.

Each leaf uses the **split-file pattern**:

```js
{ indexPath: "data/map_layers/<prefix>__index.js", indexName: "<prefix>__index", layerPrefix: "<prefix>" }
```

Exception: `mask` entries (Biodiversity GBF2 GeoJSON overlay) remain `{ path, name }`.

The index file (`window["<prefix>__index"]`) holds `{ dims: [...], tree: {…} }` where `tree` is the nested dimension hierarchy (all valid combos). Per-combo files are named `<prefix>__<safe(d1)>__…__<safe(dN)>.js`.

| Module | Category | Sub-categories (map types) |
|--------|----------|---------------------------|
| Area | Ag / Ag Mgt / Non-Ag | — |
| Biodiversity | GBF2 | Sum / Ag / Ag Mgt / Non-Ag / **mask** |
| Biodiversity | GBF3_NVIS / GBF3_IBRA / GBF4_ECNES / GBF4_SNES / GBF8_GROUP / GBF8_SPECIES / quality | Ag / Ag Mgt / Non-Ag |
| Dvar | Ag / Ag Mgt / Non-Ag / Lumap | — |
| Economics | Sum | Profit |
| Economics | Ag | Profit / Revenue / Cost / Transition(Ag2Ag) / Transition(NonAg2Ag) |
| Economics | Ag Mgt | Profit / Revenue / Cost |
| Economics | Non-Ag | Profit / Revenue / Cost / Transition(Ag2NonAg) / Transition(NonAg2NonAg) |
| GHG | Sum / Ag / Ag Mgt / Non-Ag | — |
| Production | Sum / Ag / Ag Mgt / Non-Ag | — |
| Water | Sum / Ag / Ag Mgt / Non-Ag | — |
| Renewable | Ag Mgt | `map_renewable_energy_Am.js` |
| Transition | Ag2Ag | `map_transition_area_ag2ag.js` |

### ChartService (`services/ChartService.js`)
Registry of all chart/heatmap data JS files, keyed by `chartCategories[module][category][subcategory]`.

| Module | Category | Sub-categories |
|--------|----------|---------------|
| Area | Ag / Ag Mgt / Non-Ag | — |
| Area | overview | Land-use / Category / Source |
| Area | ranking | — |
| Biodiversity | GBF2 / GBF3_NVIS / GBF3_IBRA / GBF4_SNES / GBF4_ECNES / GBF8_GROUP / GBF8_SPECIES / quality | Ag / Ag Mgt / Non-Ag |
| Biodiversity | GBF2…quality | overview → Ag / Am / NonAg / sum |
| Biodiversity | ranking | — |
| Economics | Sum | Profit |
| Economics | Ag | Profit / Revenue / Cost / Transition(Ag2Ag) / Transition(NonAg2Ag) |
| Economics | Ag Mgt | Profit / Revenue / Cost |
| Economics | Non-Ag | Profit / Revenue / Cost / Transition(Ag2NonAg) / Transition(NonAg2NonAg) |
| Economics | overview | Ag / Am / NonAg / sum |
| Economics | ranking | — |
| GHG | Sum / Ag / Ag Mgt / Non-Ag | — |
| GHG | overview | Ag / Am / NonAg / sum |
| GHG | ranking | — |
| Production | Sum / Ag / Ag Mgt / Non-Ag | — |
| Production | overview | achieve / Domestic / Exports / Feed / Imports / sum |
| Production | ranking | — |
| Water | **NRM** | Sum / Ag / Ag Mgt / Non-Ag |
| Water | NRM overview | Ag / Am / NonAg / sum |
| Water | NRM ranking | — |
| Water | **Watershed** | overview |
| Renewable | Ag Mgt | `Renewable_energy_Am.js` |
| Transition | start_end | `Transition_start_end_area.js` |
| Transition | Ag2Ag | `Transition_ag2ag_area.js` |
| Supporting | info | `Supporting_info.js` |

### MemoryService (`services/MemoryService.js`)
Prevents stale `window[...]` globals accumulating when navigating between views.

- **`registerViewData(viewName, dataNames[])`** — records which window globals belong to a view
- **`registerViewScript(viewName, src, scriptElement)`** — tracks injected `<script>` DOM nodes
- **`cleanupViewData(viewName)`** — deletes all registered globals and removes script elements
- **`getMemoryInfo()`** — returns a debug snapshot of current registry state

**Usage**: every view calls `onUnmounted(() => window.MemoryService.cleanupViewData(VIEW_NAME))`.  
`loadScriptWithTracking()` in `helpers.js` calls `registerViewData` / `registerViewScript` automatically.

---

## Component Architecture

### `<chart-container>` (`chart_container.js`)
Highcharts Vue wrapper for time-series charts.  
Props: standard Highcharts series/options structure. Manages chart lifecycle (create, update, destroy).

### `<heatmap-container>` (`heatmap_container.js`)
Highcharts heatmap Vue wrapper for From→To land-use transition matrices.  
Props: `xCats`, `yCats`, `data`, `maxVal`, `nullColor`, `showAxisLabels`, `showDataLabels`, `onCellClick`, `exportable`, `zoomable`, `draggable`.  
Accepts the `{x_categories, y_categories, data, max_val}` leaf format produced by `create_report_data.py`.

### `<regions-map>` (`regions_map.js`)
NRM region selector combining a `<map-geojson>` choropleth with a base64-image overlay.  
Emits region selection events consumed by all analysis views via `globalSelectedRegion`.

### `<map-geojson>` (`map_geojson.js`)
Raw Leaflet NRM choropleth with hover tooltip.  
Used inside `<regions-map>` and directly in views that need a standalone spatial layer.

### `<side-bar>` (`sidebar.js`)
Collapsible navigation sidebar with LUTO branding. Provides `isCollapsed` toggle.

### `<ranking-cards>` (`ranking_cards.js`)
Displays top-N land use items as sorted card list. Used on Home view.

### `<filterable-dropdown>` (`filterable_dropdown.js`)
Searchable dropdown for long option lists (e.g. land use names). Wraps Element Plus select with filter.

---

## Data Architecture

### Leaf Data Formats

**Chart leaf** (time-series) — array of Highcharts series objects:
```js
[{ name: "Beef - natural land", data: [[2020, 1.2e6], [2025, 1.1e6], ...] }, ...]
```

**Map leaf** — single-year spatial raster:
```js
{ img_str: "<base64 PNG>", bounds: [[lat0,lon0],[lat1,lon1]], min_max: [0, 12345.6] }
```

**Heatmap leaf** (transition matrix) — used by `<heatmap-container>`:
```js
{ x_categories: ["Beef - modified", ...], y_categories: [...], data: [[row,col,val], ...], max_val: 9876 }
```

### Chart Data Hierarchies

| Module | Full path to chart leaf |
|--------|------------------------|
| Area (Ag) | `data[region][lm][lu]` → `[series]` |
| Area (Am) | `data[region][lm][lu]` → `[series(name=AgMgt)]` |
| Area (NonAg) | `data[region]` → `[series(name=LU)]` |
| Economics (Ag Profit) | `data[region][lm][lu]` → `[series]` |
| Economics (Ag Revenue/Cost) | `data[region][lm][source][lu]` → `[series]` |
| Economics (Am) | `data[region][lm][lu]` → `[series(name=AgMgt)]` |
| Economics (NonAg) | `data[region]` → `[series(name=LU)]` |
| Economics (Transitions) | `data[region]` → `[series]` |
| GHG (Ag) | `data[region][lm][source][lu]` → `[series]` |
| GHG (Am) | `data[region][lm][lu]` → `[series(name=AgMgt)]` |
| GHG (NonAg) | `data[region]` → `[series(name=LU)]` |
| Water (Ag/Am/NonAg) | `data[region][lm][lu]` — Water chart series by AgMgt for Am |
| Production (Ag/Am/NonAg) | same pattern as GHG |
| Biodiversity | same pattern as Area per GBF metric |
| Renewable (Am) | `data[region][agMgt][lm]` → `[series(name=LU)]` |
| Transition heatmap | `data[region][from_water][to_water][year]` → `{x_categories, y_categories, data, max_val}` |

### Map Data Hierarchies

| Module | Full path to map leaf |
|--------|----------------------|
| Area (Ag/Am) | `data[lm][lu][year]` → leaf |
| Area (NonAg) | `data[lu][year]` → leaf |
| Economics (Ag Profit) | `data[lm][lu][year]` → leaf |
| Economics (Ag Revenue/Cost) | `data[lm][source][lu][year]` → leaf |
| Economics (Am) | `data[am][lm][lu][year]` → leaf |
| Economics (NonAg) | `data[lu][year]` → leaf |
| GHG (Ag) | `data[lm][source][lu][year]` → leaf |
| GHG (Am) | `data[am][lm][lu][year]` → leaf |
| GHG (NonAg) | `data[lu][year]` → leaf |
| Biodiversity | same pattern as Area per GBF metric |
| Dvar (Ag/Am) | `data[lm][lu][year]` → leaf (categorical integer colormap) |
| Renewable (Am) | `data[agMgt][lm][lu][year]` → leaf |
| Transition (Ag2Ag) | `data[from_water][to_water][from_lu][to_lu][year]` → leaf |

**Note on `source` dimension**: appears only in **Ag** for GHG (emission type) and Economics (cost/revenue type). Am and NonAg do not have a source level.

### Special Module Notes

- **Economics `Sum` category**: chart data contains `[series]` with `{name, type: 'Profit/Revenue/Cost', data: [[year, val], ...]}` — no water/LU nesting
- **Economics Transition types**: `Ag2Ag`, `NonAg2Ag` sit under the `Ag` category; `Ag2NonAg`, `NonAg2NonAg` under `Non-Ag`; both treat lm as irrelevant (always `ALL`)
- **Biodiversity metric gating**: `BiodiversityView` reads `window.SupportingInfo.settings` and hides any metric whose setting value is `'off'`; `quality` metric is always shown
- **Water NRM vs Watershed**: `WaterView` has two tabs — NRM (full progressive selection) and Watershed (overview chart only, no map)
- **Transition SubCat auto-discovery**: `Transition.js` derives its SubCat buttons from `Object.keys(MapService.mapCategories['Transition'])` — adding a new transition type requires only a `MapService.js` + `ChartService.js` entry; `Transition.js` is never edited. The `start_end` SubCat loads a chart-only heatmap (no map).

## Setup and Usage

### Prerequisites
A modern browser with JavaScript enabled. No server, build step, or internet connection required.

### Opening the Report
After a LUTO simulation run completes, the report is at:

```
<run_output_dir>/DATA_REPORT/index.html
```

Double-click `index.html` or drag it into a browser window. All libraries are in `lib/`.

### Regenerating Data Files
If you need to convert raw JSON output back into JS data files:

```bash
python dataTransform/01_JSON2JS_dataTrans.py
```

This wraps each JSON file as a `window.XxxName = {...}` assignment and writes it to `data/`.

---

## Development Notes

### Architecture Patterns

**No build step**: views are plain `window.XxxView = { name, setup() {...} }` objects registered in `index.js` via `app.component(...)`. No SFC, no Vite, no webpack.

**Split-file map loading**: map layer data is split into one JS file per dimension-combo plus an `__index.js` that lists valid combos. Views use `createMapLayerLoader(VIEW_NAME)` (from `helpers.js`) to get `{ currentLayerData, ensureComboLayer }`. On every selection change a watcher calls `await ensureComboLayer(layerPrefix, [dim1, …, dimN])`, which loads only the file for the chosen combo and releases the previous one for GC. `selectMapData` is always `computed(() => currentLayerData.value?.[selectYear.value] ?? {})`.

**Index-driven option lists**: on category switch, views call `ensureIndexLoaded(cat)` to lazy-load `<prefix>__index.js`, then `getTree(cat)` reads `window[indexName].tree` to get the available dimension values.

**MemoryService lifecycle**: every view follows:

```js
onMounted(async () => {
  await ensureIndexLoaded(initCat);
  const tree = getTree(initCat);
  // populate available* from tree, then:
  await ensureComboLayer(mapRegister[initCat].layerPrefix, [/* initial combo */]);
});
onUnmounted(() => window.MemoryService.cleanupViewData(VIEW_NAME));
```

**Cascade watchers**: each view has a chain of `watch` blocks that repopulate downstream option arrays when an upstream selection changes, always ending with `ensureComboLayer`. Previous selections are stored in a `previousSelections` ref keyed by category.

**Safe property access**: all data reads use optional chaining (`data?.[a]?.[b]?.[c]`) and fall back to `[]` or `null` to prevent white-screen errors on missing data.

### Adding a New Transition Type
The `Transition.js` view auto-discovers SubCat buttons from `MapService.mapCategories['Transition']`. To add a new metric type (e.g., Ag2Ag cost):
1. Write the map layer in `write.py`
2. Create the map data file and register it in `MapService.js` under `Transition → <NewSubCat>`
3. Create the heatmap chart data file and register it in `ChartService.js` under `Transition → <NewSubCat>`
4. **Do not edit `Transition.js`** — the new SubCat button appears automatically.

See `jinzhu_inspect_code/transition_view_guide.md` for a step-by-step worked example.

### Adding a New Analysis View
1. Create `views/NewView.js` following the progressive selection pattern from `Area.js`
2. Register the route in `routes/route.js`
3. Add a sidebar entry in `sidebar.js`
4. Register chart data in `ChartService.js` and map data in `MapService.js`
5. Ensure `onUnmounted` calls `MemoryService.cleanupViewData(VIEW_NAME)`

---

## Version History

### Current State
- **11 routes**: Home, Area, Economics, GHG, Water, Production, Biodiversity, Renewable, Transition, Map (DVAR), Settings
- **Transition View**: dual heatmap + spatial map; SubCats auto-discovered from MapService registry; cell-click cross-filter; `nullMessage` on NaN cells
- **Renewable Energy View**: MWh by AgMgt/Water/Landuse; `Existing Capacity` land use shows pre-simulation installations
- **Biodiversity View**: 8 metrics (quality + GBF2/3/4/8); metric visibility gated by model settings; `mask` map layer for GBF2 priority cells
- **Water View**: NRM tab (full progressive selection) + Watershed tab (overview only)
- **Economics View**: Sum/Ag/Ag Mgt/Non-Ag categories; MapType level (Profit/Revenue/Cost); Transition sub-types under Ag and Non-Ag
- **`<heatmap-container>`**: new Highcharts heatmap Vue wrapper for transition matrices
- **MemoryService**: per-view JS global lifecycle management via `cleanupViewData()`
- **All libraries offline**: dashboard opens directly from `index.html`, no server needed
