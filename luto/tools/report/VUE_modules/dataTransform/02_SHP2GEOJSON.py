import json
import geopandas as gpd
import pandas as pd
from io import BytesIO
from shapely.ops import unary_union

NRM_AUS = gpd.read_file('luto/tools/report/VUE_modules/assets/NRM_SIMPLIFY_FILTER/NRM_AUS_SIMPLIFIED.shp')
NRM_AUS_crs = NRM_AUS.crs
NRM_AUS = NRM_AUS.dissolve(by='NRM_REGION')[['geometry']].reset_index()

# Reproject to EPSG:4326 (WGS84 lat/lng) for Leaflet compatibility
if NRM_AUS.crs.to_epsg() != 4326:
    NRM_AUS = NRM_AUS.to_crs('EPSG:4326')
    
with BytesIO() as geojson_bytes:
    NRM_AUS.to_file(geojson_bytes, driver='GeoJSON')
    geojson_bytes.seek(0)
    geojson_str = json.loads(geojson_bytes.getvalue().decode('utf-8'))
    
with open('luto/tools/report/VUE_modules/data/geo/NRM_AUS.js', 'w', encoding='utf-8') as f:
    f.write(f'window.NRM_AUS = {json.dumps(geojson_str, indent=2)};\n')


# Save centroids and bounding box of NRM to JS object
NRM_AUS.loc[len(NRM_AUS)] = ['AUSTRALIA', unary_union(NRM_AUS.geometry.values)]
NRM_AUS = NRM_AUS.set_crs(NRM_AUS_crs, allow_override=True)
NRM_AUS['centroid'] = NRM_AUS.geometry.centroid.apply(lambda p: [p.y, p.x])
NRM_AUS['bounding_box'] = NRM_AUS.geometry.bounds.values.tolist()
centroid_bbox = NRM_AUS.set_index('NRM_REGION')[['centroid', 'bounding_box']].to_dict(orient='index')

with open('luto/tools/report/VUE_modules/data/geo/NRM_AUS_centroid_bbox.js', 'w', encoding='utf-8') as f:
    f.write(f'window.NRM_AUS_centroid_bbox = {json.dumps(centroid_bbox, indent=2)};\n')



# Save AUSTRALIA STATE to JS object
AUS_STATE = gpd.read_file('luto/tools/report/VUE_modules/assets/AUS_STATE_SIMPLIFIED/STE11aAust_mercator_simplified.shp')
AUS_STATE = AUS_STATE.dissolve(by='STATE_NAME')[['geometry']].reset_index()
AUS_STATE_crs = AUS_STATE.crs  # Save before reprojection

# Reproject to EPSG:4326 (WGS84 lat/lng) for Leaflet compatibility
if AUS_STATE.crs.to_epsg() != 4326:
    AUS_STATE = AUS_STATE.to_crs('EPSG:4326')

with BytesIO() as geojson_bytes:
    AUS_STATE.to_file(geojson_bytes, driver='GeoJSON')
    geojson_bytes.seek(0)
    geojson_str = json.loads(geojson_bytes.getvalue().decode('utf-8'))

with open('luto/tools/report/VUE_modules/data/geo/AUS_STATE.js', 'w', encoding='utf-8') as f:
    f.write(f'window.AUS_STATE = {json.dumps(geojson_str, indent=2)};\n')


# Save centroids and bounding box of STATE to JS object
AUS_STATE.loc[len(AUS_STATE)] = ['AUSTRALIA', unary_union(AUS_STATE.geometry.values)]
AUS_STATE = AUS_STATE.set_crs(AUS_STATE_crs, allow_override=True)
AUS_STATE['centroid'] = AUS_STATE.geometry.centroid.apply(lambda p: [p.y, p.x])
AUS_STATE['bounding_box'] = AUS_STATE.geometry.bounds.values.tolist()
centroid_bbox = AUS_STATE.drop_duplicates(subset='STATE_NAME').set_index('STATE_NAME')[['centroid', 'bounding_box']].to_dict(orient='index')

with open('luto/tools/report/VUE_modules/data/geo/AUS_STATE_centroid_bbox.js', 'w', encoding='utf-8') as f:
    f.write(f'window.AUS_STATE_centroid_bbox = {json.dumps(centroid_bbox, indent=2)};\n')
    
    
    
    
    
# Save REZ to JS object
REZ = gpd.read_file('luto/tools/report/VUE_modules/assets/REZ_boundary/aemo_rez_boundaries_2025.shp')
REZ = REZ.dissolve(by='Name')[['geometry']].reset_index()
REZ_crs = REZ.crs  # Save before reprojection

# Reproject to EPSG:4326 (WGS84 lat/lng) for Leaflet compatibility
if REZ.crs.to_epsg() != 4326:
    REZ = REZ.to_crs('EPSG:4326')
    
with BytesIO() as geojson_bytes:
    REZ.to_file(geojson_bytes, driver='GeoJSON')
    geojson_bytes.seek(0)
    geojson_str = json.loads(geojson_bytes.getvalue().decode('utf-8'))

with open('luto/tools/report/VUE_modules/data/geo/REZ.js', 'w', encoding='utf-8') as f:
    f.write(f'window.RENEWABLE_REZ = {json.dumps(geojson_str, indent=2)};\n')
    
# Save centroids and bounding box of REZ to JS object
REZ.loc[len(REZ)] = ['AUSTRALIA', unary_union(REZ.geometry.values)]
REZ = REZ.set_crs(REZ_crs, allow_override=True)
REZ['centroid'] = REZ.geometry.centroid.apply(lambda p: [p.y, p.x])
REZ['bounding_box'] = REZ.geometry.bounds.values.tolist()
centroid_bbox = REZ.set_index('Name')[['centroid', 'bounding_box']].to_dict(orient='index')

with open('luto/tools/report/VUE_modules/data/geo/REZ_centroid_bbox.js', 'w', encoding='utf-8') as f:
    f.write(f'window.RENEWABLE_REZ_centroid_bbox = {json.dumps(centroid_bbox, indent=2)};\n')




# ── Water regions (BoM GeoFabric V3.2, HR Regions): river regions and drainage divisions ──────────────────────
#
# The model sets its water limits per water region (settings.WATER_REGION_DEF: 'River Region' or 'Drainage
# Division'), so the report draws them as a region level of their own. Two steps, as for the NRM regions:
#   1. the GeoFabric layers -> a simplified shapefile each, in assets/Water_river_region and
#      assets/Water_drainage_region (run once; delete the folder to rebuild it from the GeoFabric);
#   2. the asset shapefiles -> data/geo/WATER_RIVREG.js and WATER_DRAINDIV.js, with their centroid / bounding box.
#
# The regions are named as the MODEL names them (the NAME of input/water_region_rivreg.nc and
# water_region_draindiv.nc), joined on the region id the zones script gave them, because the GeoFabric spells
# two drainage divisions differently ('Tanami -Timor Sea Coast', 'South East Coast (VICTORIA)'): the report
# looks a region's chart up by its name.
import os
import shapely
import xarray as xr

ASSETS = 'luto/tools/report/VUE_modules/assets'
GEO_DIR = 'luto/tools/report/VUE_modules/data/geo'
DATA_MASTER = 'N:/Data-Master' if os.name == 'nt' else '/mnt/n/Planet-A/Data-Master'      # read only
GEOFABRIC = f'{DATA_MASTER}/Water/GeoFabric_V3.2/HR_Regions_GDB_V3_2/HR_Regions_GDB/HR_Regions.gdb'

# asset folder, GeoFabric layer, model input file, JS variable; and how script_1_assemble_zones_data.py numbered
# the regions: river regions by the rank of their name (1-based), drainage divisions by their row in the layer
WATER_REGIONS = [
    ('Water_river_region', 'RiverRegion', 'input/water_region_rivreg.nc', 'WATER_RIVREG',
     lambda layer: layer['RivRegName'].map({name: rank + 1 for rank, name in enumerate(sorted(layer['RivRegName'].unique()))})),
    ('Water_drainage_region', 'AWRADrainageDivision', 'input/water_region_draindiv.nc', 'WATER_DRAINDIV',
     lambda layer: pd.Series(layer.index + 1, index=layer.index)),
]
SIMPLIFY_TOLERANCE_M = 5000          # Web Mercator metres; the shared borders of neighbours stay shared


def round_coordinates(coordinates, ndigits=4):
    if isinstance(coordinates, list):
        return [round_coordinates(item, ndigits) for item in coordinates]
    return round(coordinates, ndigits)

MIN_PART_KM2 = 100                   # islands and slivers under this are dropped (never a region's largest part)

for folder, layer_name, model_file, js_name, region_id_of in WATER_REGIONS:
    asset_path = f'{ASSETS}/{folder}/{folder}.shp'

    # 1. The simplified asset, built once from the GeoFabric
    if not os.path.exists(asset_path):
        layer = gpd.read_file(GEOFABRIC, layer=layer_name)
        with xr.open_dataset(model_file) as model_regions:
            model_names = dict(zip(model_regions['region_id'].values.tolist(), model_regions['NAME'].values.tolist()))
        layer['REGION_ID'] = region_id_of(layer).astype(int)
        layer = layer[layer['REGION_ID'].isin(model_names)]                 # a region with no cell in the model has no data
        layer['NAME'] = layer['REGION_ID'].map(model_names)
        regions = layer.dissolve(by='REGION_ID', aggfunc='first')[['NAME', 'geometry']].reset_index().to_crs('EPSG:3857')

        # Drop the small parts, then simplify the regions as ONE coverage so neighbours keep a common border
        parts = regions.explode(index_parts=False).reset_index(drop=True)
        parts['area_km2'] = parts.geometry.area / 1e6
        largest = parts.groupby('REGION_ID')['area_km2'].transform('max')
        parts = parts[(parts['area_km2'] >= MIN_PART_KM2) | (parts['area_km2'] == largest)]
        regions = parts.dissolve(by='REGION_ID', aggfunc='first')[['NAME', 'geometry']].reset_index()
        simplified = gpd.GeoSeries(shapely.coverage_simplify(regions.geometry.values, SIMPLIFY_TOLERANCE_M), index=regions.index, crs=regions.crs)
        regions = regions.set_geometry(simplified)
        regions = regions[~regions.geometry.is_empty]

        os.makedirs(os.path.dirname(asset_path), exist_ok=True)
        regions.to_file(asset_path)

    # 2. The asset -> JS (EPSG:4326 for Leaflet), and the centroid / bounding box of every region and of the country
    regions = gpd.read_file(asset_path)
    regions_crs = regions.crs
    regions_wgs = regions.to_crs('EPSG:4326')[['NAME', 'geometry']]
    geojson = json.loads(regions_wgs.to_json(drop_id=True))
    for feature in geojson['features']:                      # 4 decimal places is ~10 m: far under the simplification
        feature['geometry']['coordinates'] = round_coordinates(feature['geometry']['coordinates'])
    with open(f'{GEO_DIR}/{js_name}.js', 'w', encoding='utf-8') as f:
        f.write(f'window.{js_name} = {json.dumps(geojson, separators=(",", ":"))};\n')

    country = gpd.GeoDataFrame({'NAME': ['AUSTRALIA']}, geometry=[unary_union(regions.geometry.values)], crs=regions_crs)
    with_country = pd.concat([regions[['NAME', 'geometry']], country], ignore_index=True)
    with_country['centroid'] = with_country.geometry.centroid.to_crs('EPSG:4326').apply(lambda p: [p.y, p.x])
    with_country['bounding_box'] = with_country.geometry.to_crs('EPSG:4326').bounds.values.tolist()
    centroid_bbox = with_country.set_index('NAME')[['centroid', 'bounding_box']].to_dict(orient='index')
    with open(f'{GEO_DIR}/{js_name}_centroid_bbox.js', 'w', encoding='utf-8') as f:
        f.write(f'window.{js_name}_centroid_bbox = {json.dumps(centroid_bbox, indent=2)};\n')
# ── end of the water regions ────────────────────────────────────────────────────────────────────────────────



# Save merged NECMA + GBCMA past vegetation works to JS object.
#
# The merge and cleaning live with the source data, not here — see
# N:/Data-Master/NECMA_works/prepare_merged_vegetation.py, which must be re-run whenever
# either CMA supplies a new data drop. Only the derived GeoJSON is kept in this repo;
# the source layers are restricted (NECMA redistribution unagreed, GBCMA is OFFICIAL).
#
# Dissolved by CMA. Parcels overlap where a site was worked more than once (11.4% of the
# parcel area), and stacking semi-transparent fills renders those as dark bands. The
# undissolved per-parcel layer lives beside this one for analysis.
#
# NOT simplified, deliberately. These are riparian and revegetation parcels with a
# median area of 1.4 ha — many are strips only a few tens of metres wide. Douglas-Peucker
# collapses them into slivers well before it saves any meaningful space: at a 50 m
# tolerance the narrowest strips degenerate to lines and 4.6% of total area is lost, while
# the payload only falls from 1.6 MB to 0.9 MB. Full geometry is 53k vertices, which is
# smaller than the already-shipped AUS_STATE.js, so there is nothing to gain by trading
# fidelity for bytes here. Coordinates are rounded to 6 decimal places (~0.1 m) instead.
CMA_VEG = gpd.read_file('N:/Data-Master/NECMA_works/CMA_Vegetation_Merged/CMA_VEG_PROJECTS_dissolved.shp')

# Reproject to EPSG:4326 (WGS84 lat/lng) for Leaflet compatibility
if CMA_VEG.crs.to_epsg() != 4326:
    CMA_VEG = CMA_VEG.to_crs('EPSG:4326')


def _round_coords(obj, ndigits=6):
    if isinstance(obj, list):
        return [_round_coords(i, ndigits) for i in obj]
    if isinstance(obj, float):
        return round(obj, ndigits)
    return obj


geojson_str = json.loads(CMA_VEG.to_json(drop_id=True))
for feat in geojson_str['features']:
    feat['geometry']['coordinates'] = _round_coords(feat['geometry']['coordinates'])

with open('luto/tools/report/VUE_modules/data/geo/CMA_VEG.js', 'w', encoding='utf-8') as f:
    f.write(f'window.CMA_VEG_PROJECTS = {json.dumps(geojson_str, separators=(",", ":"))};\n')
