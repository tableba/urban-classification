import numpy as np
import torch
from skimage.segmentation import slic as _skimage_slic


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

def _slic_labels(
    image: np.ndarray,
    n_segments: int,
    compactness: float,
    max_iter: int,
    sigma: float,
    min_size: int = 20,
) -> np.ndarray:
    """
    Compute SLIC superpixel labels for a single image using skimage.

    Args:
        image:       H x W x C float32 in [0, 1]. Any number of channels (e.g. 9
                     Sentinel-2 bands) is supported via channel_axis=-1.
        n_segments:  Target number of superpixels.
        compactness: Spatial vs. spectral weight (higher → more square).
        max_iter:    K-means iterations.
        sigma:       Gaussian pre-smoothing applied before clustering.
        min_size:    Minimum allowed segment size in pixels. Translated into
                     skimage's relative `min_size_factor` based on the expected
                     segment size H*W/n_segments.

    Returns:
        labels: H x W int32 array with contiguous IDs in [0, K-1].
    """
    H, W, _ = image.shape

    expected = max((H * W) / max(n_segments, 1), 1.0)
    min_size_factor = float(min_size) / expected

    labels = _skimage_slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        max_num_iter=max_iter,
        sigma=sigma,
        channel_axis=-1,
        enforce_connectivity=True,
        min_size_factor=min_size_factor,
        start_label=0,
    )
    return labels.astype(np.int32)


# ---------------------------------------------------------------------------
# Use case 1 — Preprocessing: homogenisation
# ---------------------------------------------------------------------------

def preprocess_homogenise(
    images: torch.Tensor,
    n_segments: int = 300,
    compactness: float = 0.1,
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
    n_segments: int = 300,
    compactness: float = 0.1,
    max_iter: int = 10,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Append a binary boundary map channel to the image.

    The U-Net must be configured with in_channels=13 when using this
    (9 original bands + 3 spectral indices + 1 boundary map).

    Args:
        images:      B x 12 x H x W float tensor (9 S2 bands + 3 indices).
        n_segments:  Target superpixel count per image.
        compactness: SLIC spatial weight.
        max_iter:    SLIC iterations.
        sigma:       Pre-smoothing sigma.

    Returns:
        Augmented tensor: B x 13 x H x W.
        The 13th channel contains binary boundary map (1 at boundaries, 0 interior).
    """
    B, C, H, W = images.shape
    extra = torch.zeros(B, 1, H, W, dtype=images.dtype)

    for b in range(B):
        # Use only the original 9 S2 bands for SLIC, not the indices
        img_np = images[b, :9].permute(1, 2, 0).cpu().numpy()   # H x W x 9
        labels = _slic_labels(img_np, n_segments, compactness, max_iter, sigma)

        # Create binary boundary map: 1 where adjacent pixels have different labels
        boundary = np.zeros((H, W), dtype=np.float32)
        boundary[:-1, :] += (labels[:-1, :] != labels[1:, :]).astype(np.float32)
        boundary[1:, :]  += (labels[:-1, :] != labels[1:, :]).astype(np.float32)
        boundary[:, :-1] += (labels[:, :-1] != labels[:, 1:]).astype(np.float32)
        boundary[:, 1:]  += (labels[:, :-1] != labels[:, 1:]).astype(np.float32)
        boundary = np.clip(boundary, 0, 1)  # Ensure binary

        extra[b, 0] = torch.from_numpy(boundary)

    return torch.cat([images, extra.to(images.device)], dim=1)  # B x 13 x H x W


# ---------------------------------------------------------------------------
# Use case 3 — Postprocessing: superpixel consistency
# ---------------------------------------------------------------------------

def postprocess_consistency(
    logits: torch.Tensor,
    images: torch.Tensor,
    n_segments: int = 300,
    compactness: float = 0.1,
    max_iter: int = 10,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Enforce superpixel-level label consistency on U-Net predictions.

    For each superpixel, compute argmax predictions for all pixels in the region,
    then assign the majority-voted class to all pixels in that superpixel.
    Returns logits as one-hot encodings (hard voting, not soft).

    Args:
        logits:      B x 9 x H x W raw U-Net output.
        images:      B x 9 x H x W (or 12+) input images used to compute SLIC.
        n_segments:  Target superpixel count per image.
        compactness: SLIC spatial weight.
        max_iter:    SLIC iterations.
        sigma:       Pre-smoothing sigma.

    Returns:
        Refined logits: B x 9 x H x W (one-hot per superpixel).
        Argmax over dim=1 gives the consistent class map.
    """
    B, num_classes, H, W = logits.shape
    out = torch.zeros_like(logits)

    for b in range(B):
        # Use only first 9 channels if images have more (e.g., with spectral indices)
        img_np = images[b, :9].permute(1, 2, 0).cpu().numpy()   # H x W x 9
        log_np = logits[b].cpu().numpy()                         # 9 x H x W
        labels = _slic_labels(img_np, n_segments, compactness, max_iter, sigma)

        # Hard voting: argmax predictions per pixel, then majority per superpixel
        preds = log_np.argmax(axis=0)  # H x W class predictions

        refined = np.zeros_like(log_np)
        for sp_id in np.unique(labels):
            mask = labels == sp_id
            # Find majority class in this superpixel
            region_preds = preds[mask]
            unique_classes, counts = np.unique(region_preds, return_counts=True)
            winner = unique_classes[counts.argmax()]
            refined[winner, mask] = 1.0

        out[b] = torch.from_numpy(refined)

    return out.to(logits.device)