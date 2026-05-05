import numpy as np
import torch
from scipy.ndimage import label as sp_label
from scipy.ndimage import binary_dilation
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

def _merge_small_segments(labels: np.ndarray, min_size: int = 20) -> np.ndarray:
    """Merge segments smaller than min_size into their largest neighbour."""
    out = labels.copy()
    for sp_id in np.unique(out):
        mask = out == sp_id
        if mask.sum() >= min_size:
            continue

        dilated = binary_dilation(mask)
        neighbour_ids = np.unique(out[dilated & ~mask])
        if len(neighbour_ids) == 0:
            continue
        # merge into the largest neighbour
        largest = max(neighbour_ids, key=lambda i: (out == i).sum())
        out[mask] = largest
    return out

def _slic_labels(
    image: np.ndarray,
    n_segments: int,
    compactness: float,
    max_iter: int,
    sigma: float,
) -> np.ndarray:
    """
    Compute SLIC superpixel labels for a single image.

    Args:
        image:       H x W x C float32 in [0, 1].
        n_segments:  Target number of superpixels.
        compactness: Spatial vs. colour weight (higher → more square).
        max_iter:    K-means iterations.
        sigma:       Gaussian pre-smoothing applied before clustering.

    Returns:
        labels: H x W int32 array with contiguous IDs in [0, K-1].
    """


    H, W, C = image.shape

    if sigma > 0:
        image = np.stack(
            [gaussian_filter(image[..., c], sigma) for c in range(C)], axis=-1
        )

    # Grid-initialise cluster centres
    step = max(int(np.sqrt(H * W / n_segments)), 1)
    ys = np.arange(step // 2, H, step)
    xs = np.arange(step // 2, W, step)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    cy = grid_y.ravel().astype(float)
    cx = grid_x.ravel().astype(float)
    cy = np.clip(cy, 0, H - 1)
    cx = np.clip(cx, 0, W - 1)
    cc = image[cy.astype(int), cx.astype(int)]           # K x C
    centres = np.concatenate([cc, cy[:, None], cx[:, None]], axis=1)  # K x (C+2)
    K = len(centres)

    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    labels    = -np.ones((H, W), dtype=np.int32)
    distances = np.full((H, W), np.inf)

    for _ in range(max_iter):
        for k, centre in enumerate(centres):
            col_c = centre[:C]
            y_c, x_c = centre[C], centre[C + 1]

            y0, y1 = int(max(0, y_c - step)), int(min(H, y_c + step + 1))
            x0, x1 = int(max(0, x_c - step)), int(min(W, x_c + step + 1))

            d_col = np.sum((image[y0:y1, x0:x1] - col_c) ** 2, axis=-1)
            d_xy  = (yy[y0:y1, x0:x1] - y_c) ** 2 + (xx[y0:y1, x0:x1] - x_c) ** 2
            d     = d_col + (compactness / step) ** 2 * d_xy

            mask = d < distances[y0:y1, x0:x1]
            distances[y0:y1, x0:x1][mask] = d[mask]
            labels[y0:y1, x0:x1][mask] = k

        for k in range(K):
            region = labels == k
            if not region.any():
                continue
            centres[k, :C]  = image[region].mean(0)
            centres[k,  C]  = yy[region].mean()
            centres[k, C+1] = xx[region].mean()

    # Enforce connectivity: split disconnected fragments into new segments
    new_labels = np.full_like(labels, -1)
    new_id = 0
    for k in range(K):
        mask = labels == k
        if not mask.any():
            continue
        components, n = sp_label(mask)
        for c in range(1, n + 1):
            new_labels[components == c] = new_id
            new_id += 1

    new_labels = _merge_small_segments(new_labels, min_size=20)
    return new_labels   # H x W


# ---------------------------------------------------------------------------
# Use case 1 — Preprocessing: homogenisation
# ---------------------------------------------------------------------------

def preprocess_homogenise(
    images: torch.Tensor,
    n_segments: int = 200,
    compactness: float = 10.0,
    max_iter: int = 10,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Replace each pixel with the mean value of its superpixel.

    The U-Net receives a piecewise-constant image that still has the same
    shape (B x 9 x H x W) and the same number of input channels.

    Args:
        images:      B x 9 x H x W float tensor in [0, 1].
        n_segments:  Target superpixel count per image.
        compactness: SLIC spatial weight.
        max_iter:    SLIC iterations.
        sigma:       Pre-smoothing sigma.

    Returns:
        Homogenised tensor: B x 9 x H x W.
    """
    B, C, H, W = images.shape
    out = torch.zeros_like(images)

    for b in range(B):
        img_np = images[b].permute(1, 2, 0).cpu().numpy()   # H x W x 9
        labels = _slic_labels(img_np, n_segments, compactness, max_iter, sigma)

        img_out = np.zeros_like(img_np)
        for sp_id in np.unique(labels):
            mask = labels == sp_id
            img_out[mask] = img_np[mask].mean(axis=0)

        out[b] = torch.from_numpy(img_out).permute(2, 0, 1)

    return out.to(images.device)


# ---------------------------------------------------------------------------
# Use case 2 — Preprocessing: extra channel
# ---------------------------------------------------------------------------

def preprocess_extra_channel(
    images: torch.Tensor,
    n_segments: int = 200,
    compactness: float = 10.0,
    max_iter: int = 10,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Append a normalised superpixel-ID channel to the image.

    The U-Net must be configured with in_channels=10 when using this.

    Args:
        images:      B x 9 x H x W float tensor in [0, 1].
        n_segments:  Target superpixel count per image.
        compactness: SLIC spatial weight.
        max_iter:    SLIC iterations.
        sigma:       Pre-smoothing sigma.

    Returns:
        Augmented tensor: B x 10 x H x W.
        The 10th channel contains superpixel IDs normalised to [0, 1].
    """
    B, C, H, W = images.shape
    extra = torch.zeros(B, 1, H, W, dtype=images.dtype)

    for b in range(B):
        img_np = images[b].permute(1, 2, 0).cpu().numpy()   # H x W x 9
        labels = _slic_labels(img_np, n_segments, compactness, max_iter, sigma)

        labels_norm = labels.astype(np.float32) / max(labels.max(), 1)
        extra[b, 0] = torch.from_numpy(labels_norm)

    return torch.cat([images, extra.to(images.device)], dim=1)  # B x 10 x H x W


# ---------------------------------------------------------------------------
# Use case 3 — Postprocessing: superpixel consistency
# ---------------------------------------------------------------------------

def postprocess_consistency(
    logits: torch.Tensor,
    images: torch.Tensor,
    n_segments: int = 200,
    compactness: float = 10.0,
    max_iter: int = 10,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Enforce superpixel-level label consistency on U-Net output logits.

    For each superpixel, all pixels are assigned the class that received
    the highest total logit mass across the region (soft majority vote).
    The logits of every pixel in the superpixel are then replaced with
    a one-hot encoding of that winning class.

    Args:
        logits:      B x 9 x H x W raw U-Net output.
        images:      B x 9 x H x W input images used to compute SLIC
                     (pass the original images, not any homogenised version).
        n_segments:  Target superpixel count per image.
        compactness: SLIC spatial weight.
        max_iter:    SLIC iterations.
        sigma:       Pre-smoothing sigma.

    Returns:
        Refined logits: B x 9 x H x W.
        Argmax over dim=1 gives the consistent class map.
    """
    B, num_classes, H, W = logits.shape
    out = torch.zeros_like(logits)

    for b in range(B):
        img_np = images[b].permute(1, 2, 0).cpu().numpy()   # H x W x 9
        log_np = logits[b].cpu().numpy()                     # 9 x H x W
        labels = _slic_labels(img_np, n_segments, compactness, max_iter, sigma)

        refined = np.zeros_like(log_np)
        for sp_id in np.unique(labels):
            mask = labels == sp_id                           # H x W bool
            # Sum logits over superpixel → winning class
            region_logits = log_np[:, mask].sum(axis=1)     # num_classes
            winner = int(region_logits.argmax())
            refined[winner, mask] = 1.0                     # one-hot

        out[b] = torch.from_numpy(refined)

    return out.to(logits.device)