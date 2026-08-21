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
    geojson_str = eval(geojson_bytes.getvalue().decode('utf-8'))
    
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
    geojson_str = eval(geojson_bytes.getvalue().decode('utf-8'))

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
    geojson_str = eval(geojson_bytes.getvalue().decode('utf-8'))

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
