import numpy as np
import rasterio

def clean_data(data):
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data.astype("float32")

def normalize_bands(data, scale=0.0001):
    return data * scale

def compute_indices(data):
    # Expected band order: B2, B3, B4, B8, B11
    blue = data[0]
    green = data[1]
    red = data[2]
    nir = data[3]
    swir = data[4]

    ndvi = (nir - red) / (nir + red + 1e-6)
    ndbi = (swir - nir) / (swir + nir + 1e-6)

    features = np.stack([red, green, blue, ndvi, ndbi], axis=-1)
    return features

def filter_clouds(file_path):

    with rasterio.open(file_path) as src:
        cloud = src.read(1)

# Remove nodata if needed
    if src.nodata is not None:
        valid = cloud != src.nodata
        cloud = cloud[valid]

# Compute %
    cloud_percentage = (cloud == 1).sum() / cloud.size * 100

    print(f"Cloud cover: {cloud_percentage:.2f}%")
