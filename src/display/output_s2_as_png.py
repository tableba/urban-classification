from enum import Enum
import rasterio
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from src.dataset.read_data import ReadTifs

reader = ReadTifs()

def show_rgb(data, title, key):
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

    plt.title(f"{title}")
    plt.axis("off")
    plt.imsave(f"output/s2/tile_{key}_s2.png", rgb)


reader = ReadTifs()

for name, t, data in reader.loop_through_s2():
    show_rgb(data, name, name.split("_")[1])
