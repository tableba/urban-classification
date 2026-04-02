from enum import Enum
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os

IMAGES = os.path.expanduser("./resources")

window = rasterio.windows.Window(0, 0, 2000, 2000)

class TYPES(Enum):
    SAT_16 = 1
    SAT_23 = 2
    BUILD_16 = 3
    BUILD_23 = 4
    BUILD_NEW = 5

def extract_metadata_type(filename):
    name = filename.upper()

    if "NEWBUILDINGS" in name or "NEW_BUILDINGS" in name:
        return TYPES.BUILD_NEW
    elif "BUILDINGS" in name and "2016" in name:
        return TYPES.BUILD_16
    elif "BUILDINGS" in name and "2023" in name:
        return TYPES.BUILD_23
    elif "S2" in name and "2016" in name:
        return TYPES.SAT_16
    elif "S2" in name and "2023" in name:
        return TYPES.SAT_23
    else:
        raise ValueError(f"Unknown file: {filename}")

for root, _, files in os.walk(IMAGES):
    for file in files:
        if not file.lower().endswith(".tif"):
            continue

        filepath = os.path.join(root, file)

        try:
            file_type = extract_metadata_type(file)
            print(f"{file} -> {file_type.name}")

            with rasterio.open(filepath) as src:
                data = src.read(window=window)
                print(f"Data shape: {data.shape}")
                plt.imshow(data[0], cmap='gray')
                plt.title(f"{file_type.name} - {file}")
                plt.show()

        except ValueError as e:
            print(e)
