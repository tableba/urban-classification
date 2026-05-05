import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class SentinelDataset(Dataset):
    def __init__(self, patches, augment=True):
        self.patches = patches
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        s2, dw = self.patches[idx]
        s2 = torch.tensor(s2, dtype=torch.float32)
        dw = torch.tensor(dw, dtype=torch.long).squeeze(0)
        
        # Apply augmentation to both S2 and DW together
        if self.augment:
            # Stack for joint augmentation
            stacked = torch.cat([s2, dw.unsqueeze(0).float()], dim=0)  # (C+1) x H x W
            
            # Random horizontal flip
            if np.random.rand() < 0.5:
                stacked = torch.flip(stacked, dims=[2])
            
            # Random vertical flip
            if np.random.rand() < 0.5:
                stacked = torch.flip(stacked, dims=[1])
            
            # Random 90-degree rotation (0, 90, 180, or 270)
            if np.random.rand() < 0.5:
                k = np.random.randint(1, 4)  # 1, 2, or 3 times (90, 180, 270 degrees)
                stacked = torch.rot90(stacked, k=k, dims=(1, 2))
            
            s2 = stacked[:-1]
            dw = stacked[-1].long()
        
        return s2, dw


def clean_s2_data(data):
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data.astype("float32")

def normalize_bands(data, scale=0.0001):
    return data * scale


def calculate_index(a, b, eps=1e-8):
    """Calculate a normalized difference index: (a - b) / (a + b)."""
    return (a - b) / (a + b + eps)


def calculate_ndvi(data, red_index=2, nir_index=6, eps=1e-8):
    """Calculate NDVI using red and NIR bands."""
    red = data[red_index].astype(np.float32)
    nir = data[nir_index].astype(np.float32)
    return calculate_index(nir, red, eps)


def calculate_ndwi(data, green_index=1, nir_index=6, eps=1e-8):
    """Calculate NDWI using green and NIR bands."""
    green = data[green_index].astype(np.float32)
    nir = data[nir_index].astype(np.float32)
    return calculate_index(green, nir, eps)


def calculate_ndbi(data, nir_index=6, swir_index=8, eps=1e-8):
    """Calculate NDBI using SWIR and NIR bands."""
    nir = data[nir_index].astype(np.float32)
    swir = data[swir_index].astype(np.float32)
    return calculate_index(swir, nir, eps)


def add_spectral_indices(data):
    """Append NDVI, NDWI and NDBI as additional channels."""
    ndvi = calculate_ndvi(data)
    ndwi = calculate_ndwi(data)
    ndbi = calculate_ndbi(data)

    return np.concatenate(
        [data.astype(np.float32), ndvi[np.newaxis, ...], ndwi[np.newaxis, ...], ndbi[np.newaxis, ...]],
        axis=0,
    )


#we need to extract patches from the images, since they are too large to feed into a model directly 
def extract_patches(s2, dw, patch_size=256, stride=256):
    s2 = clean_s2_data(s2)
    _, height, width = s2.shape
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            s2_patch = s2[:, y:y+patch_size, x:x+patch_size]
            dw_patch = dw[:, y:y+patch_size, x:x+patch_size]
            yield s2_patch, dw_patch
