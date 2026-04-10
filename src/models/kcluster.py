import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import rasterio

def compute_ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-6)

def load_features(path, window=None):
    print(path)
    with rasterio.open(path) as src:
        data = src.read(window=window).astype("float32")

    if data.shape[0] < 9:
        raise ValueError("Need at least 9 bands (Sentinel-2)")

    data = data.astype("float32") * 0.0001

    blue = data[0]
    green = data[1]
    red = data[2]
    nir = data[3]
    swir = data[4]

    ndvi = (nir - red) / (nir + red + 1e-6)
    ndbi = (swir - nir) / (swir + nir + 1e-6)

    features = np.stack([red, green, blue, ndvi, ndbi], axis=-1)

    H, W, C = features.shape
    X = features.reshape(-1, C)

    return X, (H, W)


def cluster_image(X, n_clusters=4):
    model = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = model.fit_predict(X)
    
    cluster_ndvi = []

    for i in range(n_clusters):
        cluster_ndvi.append(ndvi[labels == i].mean())

    order = np.argsort(cluster_ndvi)

# remap labels
    labels_remaped = np.zeros_like(labels)
    for new_i, old_i in enumerate(order):
        labels_remaped[labels == old_i] = new_i

    return labels_remaped


def show_clusters(labels, shape):
    label_map = labels.reshape(shape)

    plt.imshow(label_map, cmap="tab20")
    plt.title("Land Cover Clusters")
    plt.axis("off")
    plt.show()


def run_kmeans(key, data):
    # Ensure enough bands
    if data.shape[0] < 9:
        return

    # Replace NaNs
    data = np.nan_to_num(data, nan=0.0)

    # Sentinel-2 bands (assuming order: B2, B3, B4, B5, B6, B7, B8, B8A, B11)
    blue  = data[0]  # B2
    green = data[1]  # B3
    red   = data[2]  # B4
    b5    = data[3]
    b6    = data[4]
    b7    = data[5]
    nir   = data[6]  # B8
    nir_a = data[7]  # B8A
    swir  = data[8]  # B11

    # Spectral indices
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndbi = (swir - nir) / (swir + nir + 1e-6)   # urban indicator
    ndwi = (green - nir) / (green + nir + 1e-6) # water indicator

    # Stack features (raw bands + indices)
    features = np.stack([
        blue, green, red,
        b5, b6, b7,
        nir, nir_a, swir,
        ndvi, ndbi, ndwi
    ], axis=-1)

    H, W, C = features.shape
    X = features.reshape(-1, C)

    # Optional: normalize (recommended for KMeans)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

    # KMeans
    model = KMeans(n_clusters=4, random_state=0, n_init=10)
    labels = model.fit_predict(X)

    label_map = labels.reshape(H, W)

    plt.imshow(label_map, cmap="tab20")
    plt.title("Clusters")
    plt.axis("off")
    plt.imsave(f"output/models/kmeans/tile_{key}_clusters.png", label_map, cmap="tab20")
    plt.close()
