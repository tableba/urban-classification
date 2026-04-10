import ee
import json
import time

ee.Initialize(project='thesis-485617')

# ==================== REGION ====================
mexico = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017") \
            .filter(ee.Filter.eq('country_na', 'Mexico')) \
            .geometry()

# ==================== LOAD TILES ====================
with open("./tiles/tile_geometries_sampled.geojson") as f:
    geojson = json.load(f)

features = geojson["features"]
num_tiles = len(features)
print("Total tiles:", num_tiles)

# ==================== PARAMETERS ====================
SCALE = 10
MAX_CLOUD_PROBABILITY = 65
START_DATE = '2023-01-01'
END_DATE = '2023-08-31'

# ==================== SENTINEL-2 SR ====================
s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
      .filterDate(START_DATE, END_DATE) \
      .filterBounds(mexico)

s2Clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
              .filterDate(START_DATE, END_DATE) \
              .filterBounds(mexico)

# ==================== CLOUD MASK FUNCTION ====================
def mask_clouds(img):
    clouds = ee.Image(img.get('cloud_mask')).select('probability')
    is_not_cloud = clouds.lt(MAX_CLOUD_PROBABILITY)
    return img.updateMask(is_not_cloud)

# ==================== JOIN CLOUD DATA ====================
join = ee.Join.saveFirst('cloud_mask')
s2_with_cloud_mask = join.apply(
    primary=s2,
    secondary=s2Clouds,
    condition=ee.Filter.equals(leftField='system:index', rightField='system:index')
)

# ==================== DYNAMIC WORLD ====================
dw_col = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
            .filterDate(START_DATE, END_DATE) \
            .filterBounds(mexico)

dw_composite = dw_col.select('label').mode()  # most frequent class
dw_composite = dw_composite.toUint8()  # integer classes 0-8

# ==================== S2 COMPOSITE ====================
def mask_clouds_img(img):
    return mask_clouds(img)

s2_composite = ee.ImageCollection(s2_with_cloud_mask) \
                   .map(mask_clouds_img) \
                   .median() \
                   .select(['B2','B3','B4','B5','B6','B7','B8', 'B8A','B11'])
# ==================== EXPORT PER TILE ====================

for i in range(num_tiles):
    geom = ee.Geometry(features[i]["geometry"])

    # --- S2 input ---
    s2_tile = s2_composite.clip(geom).unmask(0)

    ee.batch.Export.image.toDrive(
        image=s2_tile,
        description=f'tile_{i}_S2',
        folder='GEE_S2',
        fileNamePrefix=f'tile_{i}_S2',
        region=features[i]["geometry"]["coordinates"],
        scale=SCALE,
        maxPixels=1e13,
        fileFormat='GeoTIFF'
    ).start()

    # --- DW label ---
    dw_tile = dw_composite.clip(geom).unmask(0)

    ee.batch.Export.image.toDrive(
        image=dw_tile,
        description=f'tile_{i}_DW',
        folder='GEE_DW',
        fileNamePrefix=f'tile_{i}_DW',
        region=features[i]["geometry"]["coordinates"],
        scale=SCALE,
        maxPixels=1e13,
        fileFormat='GeoTIFF'
    ).start()

    print(f"Started exports for tile {i}")
    time.sleep(0.1)  # slight delay to avoid throttling

print("Export tasks started. Check GEE Task Manager to monitor progress.")
