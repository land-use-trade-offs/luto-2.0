import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
import os

# ==========================================
# 1. CONFIGURATION & EXACT PATHS
# ==========================================
OUTPUT_DIR = r'T:\GitHub\luto-2.0\input\REM\luto2_baselines'

# Vector and CSV Inputs (Corrected CSV filename)
TARGETS_CSV = r'T:\GitHub\luto-2.0\input\REM\renewable_targets-input.csv'
STATE_SHP = r'T:\GitHub\luto-2.0\luto\tools\report\VUE_modules\assets\AUS_STATE_SIMPLIFIED\STE11aAust_mercator_simplified.shp'

# Static Capacity Factor Rasters
CF_PATHS = {
    'Solar_PV': r'T:\GitHub\luto-2.0\input\REM\step_change\capacity_factor\capacity_factor_solar.tif',
    'Wind_Onshore': r'T:\GitHub\luto-2.0\input\REM\step_change\capacity_factor\capacity_factor_wind.tif'
}

# Static Transmission Loss Raster (Using 2030 proxy for all years)
DLF_PATH = r'T:\GitHub\luto-2.0\input\REM\step_change\transmission_loss\transmission_loss_SC_2030.tif'

# Model Assumptions
TECH_DENSITY = {
    'Solar_PV': 45.0,     # MW per cell
    'Wind_Onshore': 4.0   # MW per cell
}

TECH_CSV_MAP = {
    'Solar_PV': 'Utility Solar',
    'Wind_Onshore': 'Wind'
}

YEARS = [2025, 2026, 2027, 2028, 2029]

STATE_MAPPING = {
    'Australian Capital Territory': {'abbr': 'ACT', 'id': 1},
    'New South Wales': {'abbr': 'NSW', 'id': 2},
    'Northern Territory': {'abbr': 'NT', 'id': 3},
    'Queensland': {'abbr': 'QLD', 'id': 4},
    'South Australia': {'abbr': 'SA', 'id': 5},
    'Tasmania': {'abbr': 'TAS', 'id': 6},
    'Victoria': {'abbr': 'VIC', 'id': 7},
    'Western Australia': {'abbr': 'WA', 'id': 8},
    'ACT': {'abbr': 'ACT', 'id': 1}, 'NSW': {'abbr': 'NSW', 'id': 2},
    'NT': {'abbr': 'NT', 'id': 3}, 'QLD': {'abbr': 'QLD', 'id': 4},
    'SA': {'abbr': 'SA', 'id': 5}, 'TAS': {'abbr': 'TAS', 'id': 6},
    'VIC': {'abbr': 'VIC', 'id': 7}, 'WA': {'abbr': 'WA', 'id': 8}
}

# ==========================================
# 2. STATE RASTERIZATION
# ==========================================
print("Loading and preparing State Boundaries...")
states = gpd.read_file(STATE_SHP)
states = states.dissolve(by='STATE_NAME').reset_index()

if states.crs.to_epsg() != 4283:
    print("Reprojecting states to EPSG:4283 (GDA94)...")
    states = states.to_crs('EPSG:4283')

states['STATE_ID'] = states['STATE_NAME'].apply(lambda x: STATE_MAPPING.get(x, {}).get('id', 0))
states['CSV_ABBR'] = states['STATE_NAME'].apply(lambda x: STATE_MAPPING.get(x, {}).get('abbr', 'UNKNOWN'))

# Grab spatial metadata from a baseline raster to serve as our master template
ref_raster_path = os.path.join(OUTPUT_DIR, f"LUTO2_Baseline_Solar_PV_2025.tif")
with rasterio.open(ref_raster_path) as ref:
    meta = ref.meta.copy()
    dst_transform = ref.transform
    dst_crs = ref.crs
    dst_shape = (ref.height, ref.width)

print("Rasterizing state mask...")
shapes = ((geom, val) for geom, val in zip(states.geometry, states['STATE_ID']) if val != 0)
state_array = rasterize(shapes=shapes, out_shape=dst_shape, transform=dst_transform, fill=0, dtype=np.uint8)

# ==========================================
# 3. PRE-PROCESS HIGH-RES ARRAYS (DLF)
# ==========================================
print("Reprojecting high-resolution Transmission Loss (DLF) array...")
dlf_array = np.zeros(dst_shape, dtype=np.float32)

with rasterio.open(DLF_PATH) as src:
    reproject(
        source=rasterio.band(src, 1),
        destination=dlf_array,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.average
    )

# ==========================================
# 4. GENERATION MATH & DEDUCTION
# ==========================================
print("Loading exogenous targets CSV...")
targets_df = pd.read_csv(TARGETS_CSV)

for tech, max_mw in TECH_DENSITY.items():
    print(f"\n====================================")
    print(f"Reprojecting high-resolution Capacity Factor for {tech}...")
    
    # Pre-process the CF array for this technology once
    cf_array = np.zeros(dst_shape, dtype=np.float32)
    with rasterio.open(CF_PATHS[tech]) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=cf_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.average
        )
        
    csv_tech_name = TECH_CSV_MAP[tech]
    
    for year in YEARS:
        print(f"\n--- Processing {year} | {tech} ---")
        
        baseline_path = os.path.join(OUTPUT_DIR, f"LUTO2_Baseline_{tech}_{year}.tif")
        if not os.path.exists(baseline_path):
            print(f"  [Skip] Missing baseline raster: LUTO2_Baseline_{tech}_{year}.tif")
            continue
            
        with rasterio.open(baseline_path) as src: 
            baseline_mw = src.read(1)
            
        # Math: Gen (TWh) = [MW * CF * 8760 * (1 - DLF)] / 1,000,000
        gen_mwh = baseline_mw * cf_array * 8760.0 * (1.0 - dlf_array)
        gen_twh = gen_mwh / 1000000.0
        
        for idx, row in states.iterrows():
            state_id = row['STATE_ID']
            state_abbr = row['CSV_ABBR']
            
            if state_id == 0: continue
            
            mask = (state_array == state_id)
            state_baseline_twh = np.sum(gen_twh[mask])
            
            if state_baseline_twh > 0:
                row_mask = (targets_df['state'] == state_abbr) & (targets_df['tech'] == csv_tech_name)
                
                new_target = targets_df.loc[row_mask, str(year)] - state_baseline_twh
                targets_df.loc[row_mask, str(year)] = new_target.clip(lower=0)
                
                print(f"  {state_abbr}: Deducted {state_baseline_twh:.4f} TWh from targets")

        # Calculate and Export Headroom Raster
        headroom_array = np.clip(max_mw - baseline_mw, a_min=0, a_max=max_mw)
        headroom_out = os.path.join(OUTPUT_DIR, f"Headroom_{tech}_{year}.tif")
        
        with rasterio.open(headroom_out, 'w', **meta) as dst:
            dst.write(headroom_array, 1)

# ==========================================
# 5. SAVE RESIDUAL TARGETS
# ==========================================
out_csv = r'T:\GitHub\luto-2.0\input\REM\residual_targets_input.csv'
targets_df.to_csv(out_csv, index=False)
print(f"\n✅ Processing complete. Residual targets saved to:\n   {out_csv}")
print(f"✅ Headroom rasters generated in:\n   {OUTPUT_DIR}")