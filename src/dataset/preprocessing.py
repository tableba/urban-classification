import numpy as np
import rasterio

def clean_s2_data(data):
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data.astype("float32")

def normalize_bands(data, scale=0.0001):
    return data * scale

