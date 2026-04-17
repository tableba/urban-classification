import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

class SentinelDataset(Dataset):
    def __init__(self, patches):
        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        s2, dw = self.patches[idx]
        s2 = torch.tensor(s2, dtype=torch.float32)
        dw = torch.tensor(dw, dtype=torch.long).squeeze(0)  # Add channel dimension for DW
        return s2, dw


def clean_s2_data(data):
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data.astype("float32")

def normalize_bands(data, scale=0.0001):
    return data * scale

#we need to extract patches from the images, since they are too large to feed into a model directly 
def extract_patches(s2, dw, patch_size=256, stride=256):
    s2 = clean_s2_data(s2)
    _, height, width = s2.shape
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            s2_patch = s2[:, y:y+patch_size, x:x+patch_size]
            dw_patch = dw[:, y:y+patch_size, x:x+patch_size]
            yield s2_patch, dw_patch
