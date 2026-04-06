from enum import Enum
import rasterio
import matplotlib.pyplot as plt
import os
import re
import numpy as np

IMAGES_DIR = os.path.expanduser("./resources")
WINDOW = rasterio.windows.Window(0, 0, 2000, 2000)

class FileType(Enum):
    SAT_2016 = 1
    SAT_2023 = 2
    BUILD_2016 = 3
    BUILD_2023 = 4
    BUILD_NEW = 5

RULES = [
    (re.compile(r"NEW[_]?BUILDINGS", re.I), FileType.BUILD_NEW),
    (re.compile(r"BUILDINGS.*2016", re.I), FileType.BUILD_2016),
    (re.compile(r"BUILDINGS.*2023", re.I), FileType.BUILD_2023),
    (re.compile(r"S2.*2016", re.I), FileType.SAT_2016),
    (re.compile(r"S2.*2023", re.I), FileType.SAT_2023),
]

def get_type(filename):
    for pattern, t in RULES:
        if pattern.search(filename):
            return t
    raise ValueError(f"Unknown file: {filename}")

def get_tif_files(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".tif"):
                yield os.path.join(root, f)

def read_preview(path):
    with rasterio.open(path) as src:
        return src.read(window=WINDOW)

def show_rgb(data, title):
    if data.shape[0] < 3:
        return

    rgb = np.stack([data[2], data[1], data[0]], axis=-1).astype("float32")
    rgb = np.nan_to_num(rgb, nan=0.0)

    valid = rgb > 0
    if not np.any(valid):
        return

    p2 = np.percentile(rgb[valid], 2)
    p98 = np.percentile(rgb[valid], 98)

    rgb = (rgb - p2) / (p98 - p2 + 1e-6)
    rgb = np.clip(rgb, 0, 1)

    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()

for path in get_tif_files(IMAGES_DIR):
    name = os.path.basename(path)
    try:
        t = get_type(name)
        data = read_preview(path)

    except ValueError as e:
        print(e)
