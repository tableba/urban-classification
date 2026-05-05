import numpy as np
from skimage.segmentation import find_boundaries
from read_data import ReadTifs
from preprocessing import extract_patches, normalize_bands
from SLIC import _slic_labels, _merge_small_segments

PATCHES_PER_TILE = 5   # sample this many patches per tile
n_values = [100, 150, 200, 300, 400, 500]
results = {n: [] for n in n_values}

reader = ReadTifs()
tile_count = 0

for key, s2, dw in reader.loop_through_files():
    if tile_count >= 10:
        break
    s2 = normalize_bands(s2)
    patches = list(extract_patches(s2, dw))
    sampled = patches[::max(1, len(patches) // PATCHES_PER_TILE)][:PATCHES_PER_TILE]

    for s2_patch, dw_patch in sampled:
        img_np = s2_patch.transpose(1, 2, 0)
        dw_np  = dw_patch[0]
        dw_bounds = find_boundaries(dw_np, mode='outer')

        for n in n_values:
            labels = _slic_labels(img_np, n_segments=n, compactness=0.1, max_iter=10, sigma=1.0)
            labels = _merge_small_segments(labels, min_size=20)
            actual = labels.max() + 1
            slic_bounds = find_boundaries(labels, mode='outer')
            recall = (slic_bounds & dw_bounds).sum() / max(dw_bounds.sum(), 1)
            results[n].append((recall, actual))

    tile_count += 1
    print(f"Done tile {tile_count}/10  ({len(sampled)} patches sampled)")

# Summarise and rank
print("\n--- Results ---")
rows = []
for n in n_values:
    recalls = [r for r, _ in results[n]]
    actuals = [a for _, a in results[n]]
    mean_recall = np.mean(recalls)
    mean_actual = np.mean(actuals)
    ratio = mean_actual / n
    score = mean_recall / ratio
    rows.append((n, mean_recall, mean_actual, ratio, score))

rows.sort(key=lambda r: r[4], reverse=True)

print(f"{'n_segments':>12} {'recall':>8} {'actual':>8} {'ratio':>8} {'score':>8}")
print("-" * 48)
for n, recall, actual, ratio, score in rows:
    print(f"{n:>12}  {recall:>7.3f}  {actual:>7.1f}  {ratio:>7.2f}  {score:>7.3f}")