# LUTO2 Output Structure & NetCDF Format

This document describes the output file structure, NetCDF format specifications, and data transformations for map generation.

## Output Directory Structure

Results are saved in `/output/<timestamp>/`:
- `DATA_REPORT/REPORT_HTML/index.html`: Interactive HTML dashboard
- Spatial outputs: NetCDF files (xarray format) for mapping
- Data tables: CSV files with numerical results
- Logs: Execution logs and performance metrics

## NetCDF Output Format and Structure

The model outputs spatial results as xarray datasets saved in NetCDF format, designed to support the hierarchical progressive selection pattern in the Vue.js reporting interface.

### Dimension Hierarchies

Output files follow category-specific dimension structures that determine the progressive button hierarchy in the web interface:

1. **Agricultural Land Use (Ag)**:
   - **Base hierarchy**: `lm` (land management) → `lu` (land use) → `year` → `cell`
   - **lm values**: `"ALL"`, `"dry"` (dryland), `"irr"` (irrigated)
   - **lu values**: `"ALL"`, `"Apples"`, `"Beef"`, `"Citrus"`, etc. (28 agricultural commodities)
   - **After expansion**: Contains `lu="ALL"` aggregation across all land uses

2. **Agricultural Management (Am)**:
   - **Base hierarchy**: `am` (agricultural management) → `lm` → `lu` → `year` → `cell`
   - **am values**: `"ALL"`, `"Asparagopsis taxiformis"`, `"AgTech EI"`, `"Biochar"`, etc.
   - **After expansion**: Contains `am="ALL"` and `lm="ALL"` aggregations

3. **Non-Agricultural Land Use (NonAg)**:
   - **Base hierarchy**: `lu` → `year` → `cell`
   - **lu values**: `"ALL"`, `"Environmental Plantings"`, `"Carbon Plantings"`, etc.
   - **No expansion needed** (already simplified)

4. **Special Case - GHG Emissions**:
   - **Ag hierarchy**: `lm` → `source` → `lu` → `year` → `cell`
   - **Am hierarchy**: `am` → `lm` → `source` → `lu` → `year` → `cell`
   - **source values**: `"ALL"`, `"Chemical"`, `"Crop Management"`, `"Manure"`, etc.
   - **Additional expansion**: `source="ALL"` aggregates across emission sources

5. **Special Case - Economics (Cost/Revenue)**:
   - **Cost structure**: Includes additional source dimension for cost types
   - **source values**: `"ALL"`, `"Labour cost"`, `"Area cost"`, etc.
   - **Separate files**: Cost and revenue saved as independent NetCDF files

### Mosaic Layer Generation

Mosaic layers are integer-coded categorical maps showing the dominant land use or management practice at each cell. These layers are generated **BEFORE** multiplication with data matrices (since multiplying integer indices is meaningless).

#### Agricultural (Ag) Mosaic

```python
# 1. Generate overall mosaic (argmax across both lm and lu)
ag_mosaic_all = ag_map.sum('lm').argmax(dim='lu', skipna=False).expand_dims(lm=['ALL'])

# 2. Create dryland and irrigated mosaics using boolean land management map
lm_map = data.lmmaps[yr_cal].astype(bool)  # True = irrigated, False = dryland
ag_mosaic_dry = ag_mosaic_all.where(~lm_map).drop_vars('lm').assign_coords(lm=['dry'])
ag_mosaic_irr = ag_mosaic_all.where(lm_map).drop_vars('lm').assign_coords(lm=['irr'])

# 3. Concatenate all mosaic layers and expand dimensions
ag_mosaic = xr.concat([ag_mosaic_all, ag_mosaic_dry, ag_mosaic_irr], dim='lm')
ag_mosaic = ag_mosaic.expand_dims(lu=['ALL'])  # Add lu dimension with 'ALL' label

# 4. Apply mask to filter cells with negligible agriculture
ag_mask = ag_map.sum(['lm', 'lu']) > 0.001
ag_mosaic = xr.where(ag_mask, ag_mosaic, np.nan)
```

#### Agricultural Management (Am) Mosaic

```python
# 1. Generate overall mosaic (argmax across am, summing over lm and lu)
am_mosaic_all = am_map.sum(['lm', 'lu']).argmax(dim='am', skipna=False).expand_dims(lm=['ALL'])

# 2. Create dryland/irrigated mosaics
am_mosaic_dry = am_mosaic_all.where(~lm_map).drop_vars('lm').assign_coords(lm=['dry'])
am_mosaic_irr = am_mosaic_all.where(lm_map).drop_vars('lm').assign_coords(lm=['irr'])

# 3. Concatenate lm-level mosaics
am_mosaic_lm = xr.concat([am_mosaic_all, am_mosaic_dry, am_mosaic_irr], dim='lm')

# 4. Create per-land-use mosaics (filter by land use code)
am_mosaic_lu = xr.concat([
    am_mosaic_lm.where(am_mosaic_lm == lu_code).expand_dims(lu=[lu_desc])
    for lu_code, lu_desc in data.ALLLU2DESC.items()
    if lu_code != -1  # Exclude NoData (cells outside LUTO study area)
], dim='lu')

# 5. Combine lm-level and lu-specific mosaics, then expand am dimension
am_mosaic = xr.concat([am_mosaic_lm.expand_dims(lu=['ALL']), am_mosaic_lu], dim='lu')
am_mosaic = am_mosaic.expand_dims(am=['ALL'])

# 6. Apply mask to filter cells with negligible agricultural management
am_mask = am_map.sum(['am', 'lm', 'lu']) > 0.001
am_mosaic = xr.where(am_mask, am_mosaic, np.nan)
```

#### Non-Agricultural (NonAg) Mosaic

```python
# 1. Find dominant non-agricultural land use (argmax returns 0-N indices)
nonag_mosaic = non_ag_map.argmax(dim='lu', skipna=False)  # Shape: (cell,)

# 2. Add base code offset to distinguish from agricultural codes
nonag_mosaic = nonag_mosaic + settings.NON_AGRICULTURAL_LU_BASE_CODE  # Values: 100, 101, 102, ...

# 3. Apply mask to filter cells with negligible non-agricultural land use
non_ag_mask = non_ag_map > 0.001
nonag_mosaic = xr.where(non_ag_mask, nonag_mosaic, np.nan)

# 4. Expand dimension and assign as 'ALL' label
nonag_mosaic = nonag_mosaic.expand_dims(lu=['ALL']).astype(np.float32)
```

### Appending Mosaics to xarray Outputs

After mosaic generation, they are concatenated to the main data arrays **in the same function** (`write_dvar_and_mosaic_map`):

```python
# Ag: Prepend mosaic with lu='ALL' coordinate (mosaic first, then dvar data)
ag_map_cat = xr.concat([ag_mosaic, ag_map], dim='lu')
# Final shape: (lm=['ALL', 'dry', 'irr'], lu=['ALL', 'Apples', ..., 'Beef', ...], cell)

# Am: Prepend mosaic with am='ALL' coordinate
am_map_cat = xr.concat([am_mosaic, am_map], dim='am')
# Final shape: (am=['ALL', 'Asparagopsis', ..., 'Biochar', ...], lm=['ALL', 'dry', 'irr'], lu=['ALL', 'Apples', ..., 'Beef', ...], cell)

# NonAg: Prepend mosaic with lu='ALL' coordinate
non_ag_map_cat = xr.concat([nonag_mosaic, non_ag_map], dim='lu')
# Final shape: (lu=['ALL', 'Environmental Plantings', ..., 'Carbon Plantings', ...], cell)
```

The `ALL` dimensions serve dual purposes:
1. **Aggregated data**: For data matrices, `ALL` contains summed/averaged values across the dimension
2. **Mosaic maps**: For decision variables, `ALL` contains the categorical dominant land use/management map (integer codes)

**Key Implementation Detail**: Mosaics are **prepended** (placed first) rather than appended, making `'ALL'` the first coordinate in each dimension. This improves readability when inspecting NetCDF files.

## Valid Layers Implementation Pattern

The "valid layers" pattern is a **critical memory and disk optimization** that reduces redundant dimension combinations. Instead of saving all possible dimension combinations (most of which contain only zeros), we identify and save only layers with meaningful values (|value| > 1).

### Why Valid Layers Matter

**Problem**: Without valid layer filtering, output files contain massive numbers of empty/zero layers:
- Agricultural revenue with 3 lm × 5 sources × 28 land uses = **420 layers** per year
- Most combinations have zero values (e.g., "Irrigated + Water cost + Apples" may not exist)
- Storing all combinations wastes **memory during processing** and **disk space** in NetCDF files

**Solution**: Extract valid layers from aggregated data, filter both data arrays and mosaic layers, then combine only meaningful layers.

### Implementation Pattern by Category

The pattern varies by data category based on dimension structure:

#### 1. Agricultural (Ag) - Example: Revenue/Cost/GHG

```python
# Step 1: Get valid data layers from AUS-level aggregated dataframe
# Use sorted MultiIndex for consistent ordering
valid_rev_layers = pd.MultiIndex.from_frame(
    ag_rev_jms_AUS[['lm', 'source', 'lu']]
).sort_values()

# Step 2: Stack and select valid data layers
ag_rev_valid_layers = xr_ag_rev.stack(
    layer=['lm', 'source', 'lu']
).sel(layer=valid_rev_layers)

# Step 3: Get mosaic and filter using where()
ag_mosaic = cfxr.decode_compress_to_multi_index(
    xr.open_dataset(os.path.join(path, f'xr_dvar_ag_{yr_cal}.nc')), 'layer'
)['data'].sel(lu='ALL', lm='ALL')

# Filter mosaic where data exists, then expand dimensions
ag_mosaic_rev = ag_mosaic.where(
    xr_ag_rev.sum(dim='lu').transpose('cell', ...)
).expand_dims(lu=['ALL'])  # Ag needs lu expansion

# Step 4: Stack mosaic and filter by valid layer dimensions
ag_mosaic_rev_stack = ag_mosaic_rev.stack(layer=['lm', 'source', 'lu'])
ag_mosaic_rev_stack = ag_mosaic_rev_stack.sel(
    layer=(
        ag_mosaic_rev_stack['layer']['lm'].isin(valid_rev_layers.get_level_values('lm')) &
        ag_mosaic_rev_stack['layer']['source'].isin(valid_rev_layers.get_level_values('source'))
        # Note: 'lu' is not filtered because mosaic has lu='ALL' only
    )
)

# Step 5: Combine valid data layers with filtered mosaic
valid_layers_stack = xr.concat([ag_rev_valid_layers, ag_mosaic_rev_stack], dim='layer')
save2nc(valid_layers_stack, os.path.join(path, f'xr_revenue_ag_{yr_cal}.nc'))
```

**Key points for Ag**:
- Mosaic expansion: `.expand_dims(lu=['ALL'])` - adds `lu` dimension
- Mosaic filter: Uses `lm` and `source` (NOT `lu` since mosaic only has `lu='ALL'`)
- Dimension order: `['lm', 'source', 'lu']`

#### 2. Agricultural Management (Am) - Example: Revenue/Cost

```python
# Step 1: Get valid data layers (sorted MultiIndex)
valid_layers_revenue = pd.MultiIndex.from_frame(
    revenue_am_df_AUS[['am', 'lm', 'lu']]
).sort_values()

# Step 2: Stack and select valid data layers
xr_revenue_am_stack = xr_revenue_am.stack(
    layer=['am', 'lm', 'lu']
).sel(layer=valid_layers_revenue)

# Step 3: Get mosaic and filter
am_mosaic_ds = cfxr.decode_compress_to_multi_index(
    xr.open_dataset(os.path.join(path, f'xr_dvar_am_{yr_cal}.nc')), 'layer'
)['data'].sel(am='ALL', lm='ALL', lu='ALL')

am_mosaic_rev = am_mosaic_ds.where(
    xr_revenue_am.sum('am').transpose('cell', ...)
).expand_dims('am')  # Am needs am expansion

# Step 4: Stack mosaic and filter by lm and lu (NOT am)
am_mosaic_rev_stack = am_mosaic_rev.stack(layer=['am', 'lm', 'lu'])
am_mosaic_rev_stack = am_mosaic_rev_stack.sel(
    layer=(
        am_mosaic_rev_stack['layer']['lm'].isin(valid_layers_revenue.get_level_values('lm')) &
        am_mosaic_rev_stack['layer']['lu'].isin(valid_layers_revenue.get_level_values('lu'))
        # Note: 'am' is not filtered because mosaic has am='ALL' only
    )
)

# Step 5: Combine
valid_layers_stack = xr.concat([xr_revenue_am_stack, am_mosaic_rev_stack], dim='layer')
save2nc(valid_layers_stack, os.path.join(path, f'xr_revenue_agricultural_management_{yr_cal}.nc'))
```

**Key points for Am**:
- Mosaic expansion: `.expand_dims('am')` - adds `am` dimension
- Mosaic filter: Uses `lm` and `lu` (NOT `am` since mosaic only has `am='ALL'`)
- Dimension order: `['am', 'lm', 'lu']`

#### 3. Non-Agricultural (NonAg) - Example: Revenue/Cost

```python
# Step 1: Get valid data layers (sorted MultiIndex)
valid_rev_layers = pd.MultiIndex.from_frame(
    rev_non_ag_df_AUS[['lu']]
).sort_values()

# Step 2: Stack and select valid data layers
non_ag_rev_valid_layers = xr_revenue_non_ag.stack(
    layer=['lu']
).sel(layer=valid_rev_layers)

# Step 3: Get mosaic - simply expand and stack (no filtering)
non_ag_mosaic = cfxr.decode_compress_to_multi_index(
    xr.open_dataset(os.path.join(path, f'xr_dvar_non_ag_{yr_cal}.nc')), 'layer'
)['data'].sel(lu='ALL').expand_dims('lu').stack(layer=['lu'])

# Step 4: Combine and compute
xr_revenue_non_ag_cat = xr.concat([non_ag_mosaic, non_ag_rev_valid_layers], dim='layer').compute()

xr_revenue_non_ag_cat.attrs = {
    'min': float(xr_revenue_non_ag_cat.min().values),
    'max': float(xr_revenue_non_ag_cat.max().values)
}

save2nc(xr_revenue_non_ag_cat, os.path.join(path, f'xr_revenue_non_ag_{yr_cal}.nc'))
```

**Key points for NonAg**:
- Mosaic handling: Simply `.expand_dims('lu').stack(layer=['lu'])` - no `.where()` or `.sel()` filtering needed
- Unlike Ag/Am: NonAg mosaic doesn't require conditional filtering since it has only one dimension
- Dimension order: `['lu']` (simplest hierarchy)

#### 4. Quantity/Production Outputs - Special Case

For commodity production outputs, the dimension is renamed from `'lu'` to `'Commodity'`:

```python
# Agricultural quantity
valid_ag_layers = pd.MultiIndex.from_frame(
    ag_q_mrc_df_AUS[['lm', 'Commodity']]  # Note: 'Commodity' not 'lu'
).sort_values()

ag_mosaic = cfxr.decode_compress_to_multi_index(
    xr.open_dataset(os.path.join(path, f'xr_dvar_ag_{yr_cal}.nc')), 'layer'
)['data'].sel(lu='ALL', lm='ALL').rename({'lu': 'Commodity'})

ag_mosaic_valid = ag_mosaic.where(
    ag_q_mrc.sum('Commodity').transpose('cell', ...)
).expand_dims('Commodity')  # Expand with 'Commodity' not 'lu'

# Filter by 'lm' only (not 'Commodity' since mosaic has Commodity='ALL')
ag_mosaic_stack = ag_mosaic_valid.stack(layer=['lm', 'Commodity'])
ag_mosaic_stack = ag_mosaic_stack.sel(
    layer=ag_mosaic_stack['layer']['lm'].isin(valid_ag_layers.get_level_values('lm'))
)
```

### Summary Table: Dimension Handling by Category

| Category | Mosaic Expansion | Mosaic Filter Dimensions | Data Stack Order |
|----------|-----------------|-------------------------|------------------|
| **Ag** (Revenue/Cost/GHG) | `.expand_dims(lu=['ALL'])` | `lm`, `source` | `['lm', 'source', 'lu']` |
| **Ag** (Quantity) | `.expand_dims('Commodity')` | `lm` only | `['lm', 'Commodity']` |
| **Am** (Revenue/Cost) | `.expand_dims('am')` | `lm`, `lu` | `['am', 'lm', 'lu']` |
| **Am** (Quantity) | `.expand_dims('am')` | `lm`, `Commodity` | `['am', 'lm', 'Commodity']` |
| **NonAg** (Revenue/Cost) | `.expand_dims('lu')` | **No filtering** | `['lu']` |
| **NonAg** (Quantity) | `.expand_dims('Commodity')` | **No filtering** | `['Commodity']` |

### Critical Implementation Rules

1. **Valid layers MUST be a sorted MultiIndex**, not a DataFrame:
   ```python
   # CORRECT
   valid_layers = pd.MultiIndex.from_frame(df[['lm', 'source', 'lu']]).sort_values()

   # WRONG
   valid_layers = df[['lm', 'source', 'lu']]  # This is a DataFrame, not MultiIndex
   ```

2. **NO manual chunking** - rely on automatic dask chunking for memory efficiency

3. **Filter mosaic by dimensions OTHER than the expanded dimension**:
   - If you expand `lu=['ALL']`, filter by other dims (`lm`, `source`) but NOT `lu`
   - If you expand `am`, filter by other dims (`lm`, `lu`) but NOT `am`
   - **NonAg special case**: Expand with the dimension (`lu` or `Commodity`), but **do NOT filter** - simply `.expand_dims().stack()` and combine directly

4. **The essence of valid layers**: Reduce redundant dimension combinations to save **both memory and disk space**

### Memory Optimization Note

The implementation avoids manual chunking because intermediate arrays are no larger than the source data matrices already in memory. The comment pattern in code explains:

```python
# ------------------------- Stack array, get valid layers -------------------------
'''
We do NOT manually loop through chunks to reduce memory usage.
Because the intermediate array is no larger than the in-mem 'ag_cost_df_rjms' object.
'''
```

This differs from other write functions (like `write_quantity_separate`) where manual chunking IS required because multiplication creates large intermediate arrays.

## save2nc() Function - Optimized NetCDF Export

The `save2nc()` function implements four critical optimizations:

### 1. Cell-Dimension Chunking

```python
# Chunking strategy: Keep 'cell' dimension full size, chunk other dimensions to 1
# This enables fast reads of complete spatial layers (cell is the map dimension)
encoding = {
    var_name: {
        'chunksizes': (1, 1, ..., full_cell_size),  # Other dims=1, cell=full
        'zlib': True,
        'complevel': 5
    }
}
```

### 2. Valid Layer Filtering

```python
# Calculate which dimension combinations have meaningful data
# Skip layers where abs(sum(data)) < 1e-3 (negligible values)
valid_df = in_xr.sum(['cell'], skipna=True).to_dataframe('ly_sum').query('abs(ly_sum) > 1e-3')

# For multi-index DataFrames, iterate through each dimension level
# Remove 'ALL' entries when only one other valid option exists
# Example: If lm=['ALL', 'irr'] (no dryland), drop 'ALL' since 'irr' alone is informative
for level_name in valid_df.index.names:
    other_levels = [l for l in valid_df.index.names if l != level_name]
    grouped = valid_df.groupby(level=other_levels)

    rows_to_drop = []
    for _, group_df in grouped:
        if group_df.index.get_level_values(level_name).nunique() == 2:
            mask_all = group_df.index.get_level_values(level_name) == 'ALL'
            rows_to_drop.extend(group_df[mask_all].index.tolist())

    valid_df = valid_df.drop(rows_to_drop, errors='ignore')

# Store as NetCDF attribute for fast filtering during map generation
loop_sel = valid_df.index.to_frame().to_dict('records')
xr_dataset.attrs['valid_layers'] = str(loop_sel)
```

### 3. Min/Max Attributes

```python
# Add global min and max values as attributes for legend scaling
in_xr.attrs['min_max'] = (float(in_xr.min().values), float(in_xr.max().values))
```

### 4. Simplified NetCDF Writing

```python
# Direct write without dask compute (faster for pre-chunked data)
in_xr.astype('float32').to_netcdf(save_path, encoding=encoding)
```

## JSON Output Format: Map vs Chart Data Hierarchies

The LUTO2 reporting system generates two types of JSON files with **different dimension hierarchies**:

### Map JSON Files (Spatial Layers)
Map JSON files contain base64-encoded PNG images for spatial visualization. The dimension hierarchy is:

**For Agricultural (Ag):**
- **Hierarchy**: `lm` → `lu` → `source` (if applicable) → `year`
- **Example**: `map_GHG_Ag.js`
  ```javascript
  {
    "ALL": {
      "ALL": {
        "ALL": { "2020": {...}, "2030": {...} }  // All sources, all land uses
      },
      "Apples": {
        "ALL": { "2020": {...} }  // All sources, Apples only
      }
    },
    "Dryland": {
      "Beef": {
        "Enteric Fermentation": { "2020": {...} }  // Specific source
      }
    }
  }
  ```

**For Agricultural Management (Am):**
- **Hierarchy**: `am` → `lm` → `lu` → `source` (if applicable) → `year`
- **Example**: `map_GHG_Am.js`
  ```javascript
  {
    "ALL": {
      "ALL": {
        "ALL": {
          "ALL": { "2020": {...} }  // All am, all water, all lu, all sources
        }
      }
    },
    "Asparagopsis taxiformis": {
      "Irrigated": {
        "Beef": {
          "Enteric Fermentation": { "2020": {...} }
        }
      }
    }
  }
  ```

**For Non-Agricultural (NonAg):**
- **Hierarchy**: `lu` → `year`
- **Example**: `map_area_NonAg.js`
  ```javascript
  {
    "ALL": { "2020": {...}, "2030": {...} },
    "Environmental Plantings": { "2020": {...} },
    "Carbon Plantings": { "2020": {...} }
  }
  ```

**Note**: The `source` dimension appears in map JSON for GHG emissions (emission sources like "Enteric Fermentation", "Manure") and Economics (cost/revenue types like "Labour cost", "Area cost").

### Chart JSON Files (Time Series Data)
Chart JSON files contain Highcharts series data for plotting. The dimension hierarchy is:

**For Agricultural (Ag):**
- **Hierarchy**: `region` → `lm` → `lu` (array of series)
- **Example**: `GHG_Ag.js`
  ```javascript
  {
    "AUSTRALIA": {
      "ALL": {
        "ALL": [  // Array of series, filtered by source in GHG case
          { "name": "Beef", "data": [[2020, 100], [2030, 120]], "type": "column" },
          { "name": "Dairy", "data": [[2020, 80], [2030, 90]], "type": "column" }
        ]
      },
      "Dryland": {
        "ALL": [...]
      }
    }
  }
  ```

**For Agricultural Management (Am):**
- **Hierarchy**: `region` → `lm` → `lu` → `source` (if applicable) → `am` (array of series)
- **Example**: `GHG_Am.js`
  ```javascript
  {
    "AUSTRALIA": {
      "ALL": {
        "ALL": [  // All land uses
          { "name": "Asparagopsis taxiformis", "data": [[2020, -50], [2030, -60]] },
          { "name": "Biochar", "data": [[2020, -30], [2030, -35]] }
        ]
      },
      "Irrigated": {
        "Beef": [  // Specific land use
          { "name": "Asparagopsis taxiformis", "data": [[2020, -20], [2030, -25]] }
        ]
      }
    }
  }
  ```

**For Non-Agricultural (NonAg):**
- **Hierarchy**: `region` → `lu` (array of series)
- **Example**: `GHG_NonAg.js`
  ```javascript
  {
    "AUSTRALIA": [
      { "name": "Environmental Plantings", "data": [[2020, -100], [2030, -150]] },
      { "name": "Carbon Plantings", "data": [[2020, -80], [2030, -120]] }
    ]
  }
  ```

### Key Differences Summary

| Aspect | Map JSON | Chart JSON |
|--------|----------|------------|
| **Purpose** | Spatial visualization (base64 PNG) | Time series plots |
| **Final level** | Year → image object | Array of series objects |
| **Region** | Not included (implicit) | Top-level dimension |
| **Ag hierarchy** | `lm → lu → source → year` | `region → lm → lu` (series array) |
| **Am hierarchy** | `am → lm → lu → source → year` | `region → lm → lu → source → am` (series array) |
| **NonAg hierarchy** | `lu → year` | `region → lu` (series array) |
| **Source position** | Before `year` (for GHG/Economics) | Before final series array (Am only) |

**Critical Implementation Note**: The chart JSON for Am places `source` **before the final series array** (where series are indexed by `am`), while map JSON places `source` **before year**. This reflects how the data is consumed:
- **Map**: User selects specific source to view spatial distribution
- **Chart**: User views all `am` types, optionally filtered by source in the Vue.js component

## create_report_layers.py - NetCDF to JSON Map Workflow

The `create_report_layers.py` script transforms NetCDF outputs into web-ready hierarchical JSON **map** files:

### Step 1: Load NetCDF and Filter Valid Layers

```python
# Open NetCDF file
ds = xr.open_dataset('ag_dvar_output.nc')

# Parse valid_layers attribute (saved by save2nc)
valid_layers = json.loads(ds.attrs.get('valid_layers', '[]'))

# Loop only through valid dimension combinations
for (lu, lm, year) in valid_layers:
    layer_data = ds.sel(lu=lu, lm=lm, year=year).values  # 1D cell array
```

### Step 2: Wrap 1D to 2D Geospatial Format

```python
# Reshape 1D cell array back to 2D raster using spatial index
raster_2d = np.full((n_rows, n_cols), np.nan)
raster_2d[row_indices, col_indices] = layer_data  # Map cells to spatial positions
```

### Step 3: Reproject to EPSG:3857 (Web Mercator)

```python
# Reproject from native CRS (e.g., GDA94) to Web Mercator for Leaflet compatibility
from rasterio.warp import reproject, Resampling

reprojected_array, transform = reproject(
    source=raster_2d,
    src_crs=native_crs,
    dst_crs='EPSG:3857',
    resampling=Resampling.nearest
)
```

### Step 4: Convert to 4-Band RGBA uint8

```python
# Convert float values to RGBA color using legend mapping
from luto.tools.report.data_tools import get_map_legend, hex_color_to_numeric

colors = get_map_legend()  # Returns color legend dictionary for all map types
color_hex = colors['float']['legend']  # Get float value color mapping

# Map float values to RGBA (Red, Green, Blue, Alpha)
rgba_array = apply_colormap(reprojected_array, color_hex)  # Shape: (height, width, 4), dtype: uint8
```

### Step 5: Convert RGBA Array to Base64 PNG

```python
# Use helper function to convert 4-band RGBA array to base64-encoded PNG
from luto.tools.report.data_tools import array_to_base64

result = array_to_base64(rgba_array)  # Returns {'img_str': 'data:image/png;base64,...'}
img_str = result['img_str']
```

### Step 6: Apply rename_reorder_hierarchy()

```python
# CRITICAL: Enforce consistent dimension ordering for Vue.js progressive selection
# Order: am → lm → other → lu (lu must be LAST)
from luto.tools.report.data_tools import rename_reorder_hierarchy

# Input: dimension selection dict like {'lm': 'dry', 'lu': 'Apples'}
# Output: Renamed and reordered dict with proper hierarchy
hierarchy_sel = rename_reorder_hierarchy(sel)
# Example: {'lm': 'Dryland', 'lu': 'Apples'} (renamed and ordered)
```

### Step 7: Build Hierarchical JSON Structure

```python
# Use helper function to convert flat tuple-keyed dict to nested structure
from luto.tools.report.data_tools import tuple_dict_to_nested

# Build flat dictionary with tuple keys
flat_dict = {}
for lm in ['ALL', 'dry', 'irr']:
    for lu in ['ALL', 'Apples', 'Beef', ...]:
        for year in [2020, 2030, 2050]:
            # Apply rename_reorder_hierarchy to get consistent ordering
            sel = rename_reorder_hierarchy({'lm': lm, 'lu': lu})

            # Create tuple key maintaining hierarchy order
            key = tuple(sel.values()) + (year,)
            flat_dict[key] = {
                'img_str': img_str,
                'bounds': [[lat_min, lon_min], [lat_max, lon_max]],
                'min_max': [data_min, data_max]
            }

# Convert flat dict to nested dict
map_json = tuple_dict_to_nested(flat_dict)

# Save as JSON file
with open('map_ag_dvar.json', 'w') as f:
    json.dump(map_json, f)
```

### Key Attributes in JSON Output

- **img_str**: Base64-encoded PNG string (format: `'data:image/png;base64,...'`)
- **bounds**: Geographic bounds in EPSG:3857 `[[south, west], [north, east]]`
- **min_max**: Data value range `[minimum, maximum]` for legend scaling

### Why 'lu' Must Be Last Dimension

The Vue.js reporting interface aggregates chart data at the land use (`lu`) level. By ensuring `lu` is the terminal dimension in the hierarchy, users can:
1. Select upstream dimensions (category, water, agmgt)
2. See all available land uses for that selection
3. Filter both map layers AND chart data by the same `lu` value

This maintains consistency between spatial maps and statistical charts, ensuring synchronized filtering throughout the interactive dashboard.

## Carbon Sequestration Data Format

The carbon sequestration data has been migrated from HDF5/pandas format to NetCDF/xarray format for improved performance and flexibility.

### Data Structure

- **Format**: NetCDF (.nc) files with compressed chunking
- **Dimensions**: `age` × `cell` (age dimension contains specific tree ages, cell dimension covers all spatial cells)
- **Available ages**: 50, 60, 70, 80, 90 years (selected from full timeseries)
- **Encoding**: zlib compression (level 5) with chunk size (1, 6956407)

### Carbon Components

Each NetCDF file contains three data variables representing different carbon pools:
1. **Trees**: Aboveground biomass in trees (`*_TREES_T_CO2_HA` or `*_TREES_TOT_T_CO2_HA`)
2. **Debris**: Aboveground debris/litter (`*_DEBRIS_T_CO2_HA` or `*_DEBRIS_TOT_T_CO2_HA`)
3. **Soil**: Belowground soil carbon (`*_SOIL_T_CO2_HA` or `*_SOIL_TOT_T_CO2_HA`)

### File Naming Convention

- Environmental Plantings (Block): `tCO2_ha_ep_block.nc`
- Environmental Plantings (Belt): `tCO2_ha_ep_belt.nc`
- Environmental Plantings (Riparian): `tCO2_ha_ep_rip.nc`
- Carbon Plantings (Block): `tCO2_ha_cp_block.nc`
- Carbon Plantings (Belt): `tCO2_ha_cp_belt.nc`
- Human-Induced Regeneration (Block): `tCO2_ha_hir_block.nc`
- Human-Induced Regeneration (Riparian): `tCO2_ha_hir_rip.nc`

### Data Loading Pattern

```python
import xarray as xr
# Load NetCDF and select specific age from CARBON_EFFECTS_WINDOW setting
ds = xr.open_dataset(os.path.join(settings.INPUT_DIR, "tCO2_ha_ep_block.nc"))
ds = ds.sel(age=settings.CARBON_EFFECTS_WINDOW, cell=self.MASK)

# Calculate total sequestration with risk discounting
total_co2 = (
    (ds.EP_BLOCK_TREES_TOT_T_CO2_HA + ds.EP_BLOCK_DEBRIS_TOT_T_CO2_HA)
    * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL)  # Aboveground with risk discount
    + ds.EP_BLOCK_SOIL_TOT_T_CO2_HA  # Belowground (no risk discount)
).values / settings.CARBON_EFFECTS_WINDOW  # Average over window
```

### Risk Discounting

- **Aboveground carbon** (Trees + Debris): Discounted by fire risk and reversal risk
- **Belowground carbon** (Soil): No risk discounting applied
- Formula: `(AG_carbon × fire_risk% × (1 - RISK_OF_REVERSAL)) + BG_carbon`

### Migration Notes

- **Old format**: HDF5 files with pandas DataFrames, separate AG/BG columns
- **New format**: NetCDF files with xarray Datasets, separate component variables
- **Advantages**: Better compression, faster subsetting, age dimension flexibility, xarray integration
- **CARBON_EFFECTS_WINDOW**: Must be one of [50, 60, 70, 80, 90] to match available data ages
