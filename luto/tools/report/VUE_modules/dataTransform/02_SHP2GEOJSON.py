import json
import geopandas as gpd
from io import BytesIO

# Save NRM to JS object
NRM_AUS = gpd.read_file('assets/NRM_SIMPLIFY_FILTER/NRM_AUS_SIMPLIFIED.shp')

with BytesIO() as geojson_bytes:
    NRM_AUS.to_file(geojson_bytes, driver='GeoJSON')
    geojson_bytes.seek(0)
    geojson_str = eval(geojson_bytes.getvalue().decode('utf-8'))
    
with open('data/geo/NRM_AUS.js', 'w', encoding='utf-8') as f:
    f.write(f'window.NRM_AUS = {json.dumps(geojson_str, indent=2)};\n')



# Save AUSTRALIA STATE to JS object
AUS_STATE = gpd.read_file('assets/AUS_STATE_SIMPLIFIED/STE11aAust_mercator_simplified.shp')

with BytesIO() as geojson_bytes:
    AUS_STATE.to_file(geojson_bytes, driver='GeoJSON')
    geojson_bytes.seek(0)
    geojson_str = eval(geojson_bytes.getvalue().decode('utf-8'))

with open('data/geo/AUS_STATE.js', 'w', encoding='utf-8') as f:
    f.write(f'window.AUS_STATE = {json.dumps(geojson_str, indent=2)};\n')
