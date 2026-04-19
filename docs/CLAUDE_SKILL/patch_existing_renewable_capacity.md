# Skill: Patching Existing Renewable Capacity into Output Arrays

This skill documents the canonical pattern for injecting real-world (pre-simulation) existing renewable capacity into any output xarray **before `add_all`**, so that `lu='Existing Capacity'` appears as a first-class layer in every derived CSV, NetCDF, and Vue map view without any downstream special-casing.

---

## Core Design Principle

> **Patch the result of `dvar × mat` (after multiplication), not the dvar array itself.**

The LP dvar for a cell reflects what the optimiser *allocated* — it never encodes existing (real-world) installations. Existing capacity is injected as a separate `lu='Existing Capacity'` slice directly into the **result xarray**, **before `add_all`** runs. Because `add_all` then computes ALL aggregates over `lm` and `lu`, the existing capacity is included for free in every summary. The downstream `to_region_and_aus_df` → `valid_layers` → `.sel()` chain sees a consistent xarray and DataFrame — no KeyError.

### What NOT to do
- Do not inject into the dvar array — that would alter the LP allocation record.
- Do not patch the DataFrame *after* `to_region_and_aus_df` — the xarray and DataFrame go out of sync, causing `KeyError` when `xr.stack(...).sel(layer=valid_layers)` is called.
- Do not add a separate writer function that concatenates existing capacity rows onto the CSV separately — this was the old `write_renewable_economics` approach; it is now deleted.

---

## The Reusable Injection Block

This block is the same for every output type. Copy it, substituting the per-cell value array and the target xarray.

```python
if any(settings.RENEWABLES_OPTIONS.values()):
    # 1. Get per-cell values for solar and wind
    solar_exist_vals = <get_solar_per_cell_array>   # np.ndarray or xr.DataArray[cell]
    wind_exist_vals  = <get_wind_per_cell_array>

    # 2. Build [am, cell] DataArray with lm='dry', lu='Existing Capacity'
    exist_re_dry = xr.DataArray(
        np.stack([solar_exist_vals, wind_exist_vals], axis=0),
        dims=['am', 'cell'],
        coords={
            'am':     ['Utility Solar PV', 'Onshore Wind'],
            'cell':   range(data.NCELLS),
            'region': ('cell', data.REGION_NRM_NAME),   # or REGION_STATE_NAME if needed
        },
    ).expand_dims(lm=['dry'], lu=['Existing Capacity'])

    # 3. irr is zeros — avoids double-counting when groupby sums over lm
    exist_re_irr  = xr.zeros_like(exist_re_dry).assign_coords(lm=['irr'])

    # 4. Combine lm and broadcast to all am types
    exist_re_full = (
        xr.concat([exist_re_dry, exist_re_irr], dim='lm')
        .reindex(am=<target_xarray>.am.values, fill_value=0.0)
    )

    # 5. Concat into the result xarray before add_all
    <target_xarray> = xr.concat([<target_xarray>, exist_re_full], dim='lu')

# Then add_all and to_region_and_aus_df as normal
```

**Key rules:**
- `lm='dry'` gets real values; `lm='irr'` is zeros — groupby sums then give the right `lm='ALL'`.
- `.reindex(am=..., fill_value=0.0)` keeps the lu dimension Cartesian; non-RE ams get 0 and are filtered downstream by `abs(value) > threshold`.
- Inject **before** `add_all` so that `lu='ALL'` and `lm='ALL'` aggregates include existing capacity automatically.

---

## Four Active Injection Points

### 1. `write_dvar_and_mosaic_map` → `xr_dvar_am_{yr_cal}.nc`

**Per-cell source:** `ag_quantity.get_existing_renewable_dvar_fraction(data, re_type, yr_cal)` — already returns `np.ndarray[cell]` (fraction 0–1, no `return_cells` needed).

```python
solar_exist_r = ag_quantity.get_existing_renewable_dvar_fraction(data, 'Utility Solar PV', yr_cal)
wind_exist_r  = ag_quantity.get_existing_renewable_dvar_fraction(data, 'Onshore Wind',     yr_cal)
```

Inject into `am_map` **after** `am_mask` is computed (mask must stay based on optimised dvars only) and **before** `add_all(am_map, ['lm', 'lu'])`.

> **Mosaic guard**: `am_argmax` (the categorical mosaic for `am='ALL'`) is built from `data.AGLU2DESC` only — 'Existing Capacity' is not in that dict, so `lu='Existing Capacity'` at `am='ALL'` is NaN → filtered out by `valid_layers > 0.001`. No explicit exclusion needed here.

### 2. `write_dvar_area` → area CSV + area NetCDF

**Per-cell source:** same `get_existing_renewable_dvar_fraction`, multiplied by `data.REAL_AREA`.

```python
solar_exist_r = ag_quantity.get_existing_renewable_dvar_fraction(data, 'Utility Solar PV', yr_cal)
wind_exist_r  = ag_quantity.get_existing_renewable_dvar_fraction(data, 'Onshore Wind',     yr_cal)
# values: solar_exist_r * data.REAL_AREA, wind_exist_r * data.REAL_AREA
```

Inject into `area_am` before `add_all(area_am, ['lm', 'lu'])`.

> **Mosaic guard**: `write_dvar_area` has an **explicit** filter at the am mosaic step:
> ```python
> valid_am_lu_mosaic = valid_am_layers.get_level_values('lu').difference(['Existing Capacity'])
> ```
> This prevents a float dvar fraction from being misinterpreted as a categorical lu code.

### 3. `write_economics` → `xr_economics_am_cost_{yr_cal}.nc` + cost CSV

Two-part injection, both before `xr_profit_am = xr_revenue_am - xr_cost_am`:

**Part 1 — Potential (optimised) cost** — replaces zeroed solar/wind slices with gap-CAPEX corrected values:
- Source: `ag_cost.get_utility_solar_pv_effect_c_mrj(data, ag_cost_mrj, yr_idx, aggregate=False)` → `{'opex': ndarray[lm,cell,lu], 'capex': ndarray[lm,cell,lu]}`
- Cost = `dvar_now × opex + dvar_delta × capex` where `dvar_delta = dvar_now − dvar_pre` (gap approach; charges CAPEX only on new-period installations)
- `.reindex(lu=xr_cost_am.lu.values, fill_value=0.0).expand_dims(am=[...])` then add via `reindex_like`

**Part 2 — Existing capacity cost** — inject as `lu='Existing Capacity'`:
- Source: `ag_cost.get_utility_solar_pv_existing_cost_by_region(data, yr_idx, return_cells=True)` → `{'opex_r': DataArray[cell], 'capex_r': DataArray[cell]}`
- Cost = `opex_now + capex_now − capex_pre` (gap approach for CAPEX)
- Also inject `xr.zeros_like(exist_re_cost_full)` into `xr_revenue_am` so `xr_profit_am = xr_revenue_am − xr_cost_am` sees matching lu dims.

#### `return_cells=True` parameter

`get_utility_solar_pv_existing_cost_by_region` and `get_onshore_wind_existing_cost_by_region` in `cost.py` accept `return_cells=True`, which returns the per-cell DataArrays **before** any regional groupby:

```python
# return_cells=True: bypass groupby, return per-cell xarrays
result = ag_cost.get_utility_solar_pv_existing_cost_by_region(data, yr_idx, return_cells=True)
opex_r  = result['opex_r']   # xr.DataArray[cell]
capex_r = result['capex_r']  # xr.DataArray[cell]
```

When `settings.AG_MANAGEMENTS['Utility Solar PV']` is False, the function returns `{'opex_r': zeros_r, 'capex_r': zeros_r}` — safe to call unconditionally inside the `if renewable_ams:` block.

### 4. `write_renewable_production` → `xr_renewable_energy_{yr_cal}.nc` + production CSV

**Per-cell source:** `ag_quantity.get_exist_renewable_capacity(data, re_type, yr_cal)` — returns `xr.DataArray[cell]` of MWh/year (already per-cell, no `return_cells` needed).

```python
solar_exist_mwh_r = ag_quantity.get_exist_renewable_capacity(data, 'Utility Solar PV', yr_cal)
wind_exist_mwh_r  = ag_quantity.get_exist_renewable_capacity(data, 'Onshore Wind',     yr_cal)
# values: .values (numpy) for np.stack
```

Inject into `renewable_energy` (= `am_dvar_mrj * renewable_potentials`) before `add_all(renewable_energy, ['am', 'lu', 'lm'])`.

---

## downstream Behaviour (no changes required)

After injection + `add_all`:

| Step | Behaviour |
|---|---|
| `to_region_and_aus_df(...)` | `lu='Existing Capacity'` rows appear naturally in region df and AUS df |
| `valid_layers = pd.MultiIndex.from_frame(df_AUS[['am','lm','lu']])` | includes `lu='Existing Capacity'` |
| `xr.stack(layer=[...]).sel(layer=valid_layers)` | no KeyError — xarray and df derived from same source |
| `save2nc(...)` | 'Existing Capacity' layers written to NetCDF |
| `get_map2json(...)` | `legend_int_level={'am':'ALL'}` → existing capacity layers render as **float** maps |
| Vue `availableLanduse` | built from `Object.keys(amData?.[am]?.[lm] || {})` — 'Existing Capacity' appears dynamically |

### `parameters.py` (already registered)
- `LANDUSE_ALL_RENAMED` line 146: `+ ['Existing Capacity']` — safe for `.index(x)` sort calls
- `COLORS_PLOT` line 405: `'Existing Capacity': '#d5e8eb'` — color for chart series

---

## Adding a New Output Type

If a new write function needs to surface existing capacity, follow this checklist:

1. **Find the result xarray** — the output of `dvar × mat` or `dvar × potentials`.
2. **Identify the per-cell source function** — use `get_exist_renewable_capacity` (MWh), `get_existing_renewable_dvar_fraction` (fraction), or `get_*_existing_cost_by_region(..., return_cells=True)` (AUD).
3. **Choose the region coord** — `data.REGION_NRM_NAME` for NRM-level reporting, `data.REGION_STATE_NAME` for state-level.
4. **Apply the reusable injection block** before `add_all`.
5. **Check if categorical mosaic uses lu** — if so, add an explicit `.difference(['Existing Capacity'])` filter before the argmax/mosaic selection step.
6. **No changes needed** in `create_report_data.py`, `create_report_layers.py`, or Vue components — all handle 'Existing Capacity' dynamically.
