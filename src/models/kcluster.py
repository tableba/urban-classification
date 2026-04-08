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

    if data.shape[0] < 4:
        raise ValueError("Need at least 5 bands (Sentinel-2)")

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


def run(name, data):
    if data.shape[0] < 5:
        return

    data = np.nan_to_num(data, nan=0.0)

    blue = data[1]
    green = data[2]
    red = data[3]
    nir = data[7] if data.shape[0] > 7 else data[3]

    ndvi = (nir - red) / (nir + red + 1e-6)

    features = np.stack([red, green, blue, ndvi], axis=-1)
    H, W, C = features.shape
    X = features.reshape(-1, C)

    model = KMeans(n_clusters=4, random_state=0, n_init=10)
    labels = model.fit_predict(X)

    label_map = labels.reshape(H, W)

    plt.imshow(label_map, cmap="tab20")
    plt.title("Clusters")
    plt.axis("off")

    plt.imsave(f"output/{name}_clusters.png", label_map, cmap="tab20")
