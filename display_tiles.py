import rasterio
import numpy as np
import matplotlib.pyplot as plt

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
