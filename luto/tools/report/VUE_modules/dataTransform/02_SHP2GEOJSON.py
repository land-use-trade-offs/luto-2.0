import json
import geopandas as gpd
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
AUS_STATE = AUS_STATE.dissolve(by='STATE_NAME').reset_index()

# Reproject to EPSG:4326 (WGS84 lat/lng) for Leaflet compatibility
if AUS_STATE.crs.to_epsg() != 4326:
    AUS_STATE = AUS_STATE.to_crs('EPSG:4326')

with BytesIO() as geojson_bytes:
    AUS_STATE.to_file(geojson_bytes, driver='GeoJSON')
    geojson_bytes.seek(0)
    geojson_str = eval(geojson_bytes.getvalue().decode('utf-8'))

with open('luto/tools/report/VUE_modules/data/geo/AUS_STATE.js', 'w', encoding='utf-8') as f:
    f.write(f'window.AUS_STATE = {json.dumps(geojson_str, indent=2)};\n')
