import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Paths
s2_path = "./resources/tile_289_S2_2016.tif"

window = rasterio.windows.Window(0, 0, 2000, 2000)

# --- Load Sentinel-2 ---
with rasterio.open(s2_path) as src:
    blue = src.read(1, window=window)   # B2
    green = src.read(2, window=window)  # B3
    red = src.read(3, window=window)    # B4

rgb = np.dstack((red, green, blue)).astype(float)

# Replace NaNs with 0
rgb = np.nan_to_num(rgb, nan=0.0)

# Better normalization (ignore zeros if needed)
valid = rgb > 0

p2 = np.percentile(rgb[valid], 2)
p98 = np.percentile(rgb[valid], 98)

rgb = (rgb - p2) / (p98 - p2)
rgb = np.clip(rgb, 0, 1)

plt.imshow(rgb)
plt.axis('off')
plt.savefig("tile_114.png", dpi=300, bbox_inches='tight')

