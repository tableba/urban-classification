import ee
import json
import time

# -------------------- INIT --------------------
ee.Initialize(project='thesis-485617')

# -------------------- LOAD GEOJSON --------------------
with open("./tiles/tile_geometries_remastered.geojson") as f:
    geojson = json.load(f)

features = geojson["features"]
n_tiles = len(features)
print("Total tiles:", n_tiles)

# -------------------- PARAMETERS --------------------
SCALE = 10            # meters
BATCH_SIZE = 1000
BATCH_INDEX = 0

start = BATCH_INDEX * BATCH_SIZE
end = min(start + BATCH_SIZE, n_tiles)
print(f"Processing batch {BATCH_INDEX}: {start} → {end}")

# -------------------- CLOUD MASK FUNCTION --------------------
def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask).divide(10000)

# -------------------- SATELLITE IMAGES --------------------
mexico_geom = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017") \
    .filter(ee.Filter.eq('country_na', 'Mexico')) \
    .geometry()

s2_2016 = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
           .filterDate('2016-06-01', '2016-06-30')
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
           .filterBounds(mexico_geom)
           .map(mask_s2_clouds)
           .median()
           .select(['B2','B3','B4','B8','B11']))

s2_2023 = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
           .filterDate('2023-06-01', '2023-06-30')
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
           .filterBounds(mexico_geom)
           .map(mask_s2_clouds)
           .median()
           .select(['B2','B3','B4','B8','B11']))

# -------------------- BUILDINGS --------------------
buildings = ee.ImageCollection('GOOGLE/Research/open-buildings-temporal/v1')

epoch2016 = ee.Date('2016-06-30', 'America/Los_Angeles').millis().divide(1000)
epoch2023 = ee.Date('2023-06-30', 'America/Los_Angeles').millis().divide(1000)

b2016 = buildings.filter(ee.Filter.eq('inference_time_epoch_s', epoch2016)) \
                 .filterBounds(mexico_geom).mosaic().select('building_presence').selfMask()

b2023 = buildings.filter(ee.Filter.eq('inference_time_epoch_s', epoch2023)) \
                 .filterBounds(mexico_geom).mosaic().select('building_presence').selfMask()

new_buildings = b2023.gt(0.5).And( b2016.gt(0.5).Not())
new_buildings_mask = new_buildings.rename('new_building').uint8()

# -------------------- EXPORT LOOP --------------------
for i in range(start, end):
    geom = ee.Geometry(features[i]["geometry"])

    # # --- Export new buildings mask ---
    task_buildings_new = ee.batch.Export.image.toDrive(
        image=new_buildings_mask.clip(geom).unmask(0),
        description=f"tile_{i}_NEW_BUILDINGS",
        folder="GEE_mexico_tiles",
        fileNamePrefix=f"tile_{i}_NEW_BUILDINGS",
        region=features[i]["geometry"]["coordinates"],
        scale=10,
        maxPixels=1e13,
        fileFormat='GeoTIFF'
    )
    task_buildings_new.start()

    task_buildings_2016 = ee.batch.Export.image.toDrive(
        image=b2016.clip(geom).unmask(0),
        description=f"tile_{i}_2016_BUILDINGS",
        folder="GEE_mexico_tiles",
        fileNamePrefix=f"tile_{i}_2016_BUILDINGS",
        region=features[i]["geometry"]["coordinates"],
        scale=10,
        maxPixels=1e13,
        fileFormat='GeoTIFF'
    )
    task_buildings_2016.start()

    task_buildings_2023 = ee.batch.Export.image.toDrive(
        image=b2023.clip(geom).unmask(0),
        description=f"tile_{i}_2023_BUILDINGS",
        folder="GEE_mexico_tiles",
        fileNamePrefix=f"tile_{i}_2023_BUILDINGS",
        region=features[i]["geometry"]["coordinates"],
        scale=10,
        maxPixels=1e13,
        fileFormat='GeoTIFF'
    )
    task_buildings_2023.start()

    # --- Export Sentinel-2 2016 ---
    # task_s2_2016 = ee.batch.Export.image.toDrive(
    #     image=s2_2016.clip(geom).unmask(0),
    #     description=f"tile_{i}_S2_2016",
    #     folder="GEE_mexico_tiles",
    #     fileNamePrefix=f"tile_{i}_S2_2016",
    #     region=features[i]["geometry"]["coordinates"],
    #     scale=SCALE,
    #     maxPixels=1e13,
    #     fileFormat='GeoTIFF'
    # )
    # task_s2_2016.start()

    # --- Export Sentinel-2 2023 ---
    # task_s2_2023 = ee.batch.Export.image.toDrive(
    #     image=s2_2023.clip(geom).unmask(0),
    #     description=f"tile_{i}_S2_2023",
    #     folder="GEE_mexico_tiles",
    #     fileNamePrefix=f"tile_{i}_S2_2023",
    #     region=features[i]["geometry"]["coordinates"],
    #     scale=SCALE,
    #     maxPixels=1e13,
    #     fileFormat='GeoTIFF'
    # )
    # task_s2_2023.start()
    #
    print(f"Started exports for tile {i}")
    time.sleep(0.1)  # slight delay to avoid throttling

print("Batch submitted.")
