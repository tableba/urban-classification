import numpy as np
import matplotlib.pyplot as plt
from read_data import ReadTifs
from preprocessing import extract_patches, normalize_bands
from SLIC import _slic_labels
from skimage.segmentation import mark_boundaries

# Grab the first patch from the first tile
reader = ReadTifs()
for key, s2, dw in reader.loop_through_files():
    s2 = normalize_bands(s2)                          # scale to [0, 1]
    for s2_patch, dw_patch in extract_patches(s2, dw):
        break
    break

# SLIC expects H x W x C
img_np = s2_patch.transpose(1, 2, 0)                 # 256 x 256 x 9
labels = _slic_labels(img_np, n_segments=500, compactness=0.1, max_iter=10, sigma=1.0)

# Pick 3 visible bands for display (e.g. bands 3, 2, 1 = RGB-ish for Sentinel-2)
rgb = img_np[:, :, [3, 2, 1]]
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

# Draw superpixel boundaries
bounded = mark_boundaries(rgb, labels, color=(0.5, 1, 0))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(rgb)
axes[0].set_title("Original patch (RGB)")
axes[1].imshow(bounded)
axes[1].set_title(f"SLIC boundaries ({labels.max()+1} superpixels)")
for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig("slic_example.png", dpi=150)
print("Saved slic_example.png")