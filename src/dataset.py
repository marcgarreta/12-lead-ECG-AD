import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    """
    Loads raw ECGs saved as 2D .npy files (samples × leads) from a directory,
    padding or truncating each to a fixed length.
    """
    def __init__(self, npy_folder: str, sample_length: int = 5000):
        self.npy_folder = npy_folder
        self.files = sorted(f for f in os.listdir(npy_folder) if f.endswith('.npy'))
        if not self.files:
            raise RuntimeError(f"No .npy files found in {npy_folder}")
        self.sample_length = sample_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = os.path.join(self.npy_folder, self.files[idx])
        arr = np.load(path)  # shape: (samples, leads)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        sig = torch.from_numpy(arr).float()
        if arr.ndim != 2:
            raise ValueError(f"File {self.files[idx]} has invalid shape {arr.shape}")
        sig = torch.from_numpy(arr).float()
        L, C = sig.shape

        if L >= self.sample_length:
            return sig[:self.sample_length]
        else:
            # pad with zeros at end
            pad = torch.zeros(self.sample_length - L, C, dtype=sig.dtype)
            return torch.cat([sig, pad], dim=0)

class WindowDataset(Dataset):
    """
    Wraps NumpyECGDataset to yield sliding windows along the time axis.
    """
    def __init__(
        self,
        npy_folder: str,
        window_size: int = 500,
        stride: int = 250,
        sample_length: int = 5000
    ):
        self.base = ECGDataset(npy_folder, sample_length=sample_length)
        self.window_size = window_size
        self.stride = stride
        self.indexes = []
        # Precompute mappings (file_idx, start_idx)
        for file_idx in range(len(self.base)):
            for start in range(0, sample_length - window_size + 1, stride):
                self.indexes.append((file_idx, start))

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx, start = self.indexes[idx]
        sig = self.base[file_idx]  # tensor shape: (sample_length, leads)
        window = sig[start : start + self.window_size]
        # Verify window data for NaNs or infinite values
        if torch.isnan(window).any():
            print(f"Warning: NaN values found in window from file {self.base.files[file_idx]}, start {start}")
        if torch.isinf(window).any():
            print(f"Warning: infinite values found in window from file {self.base.files[file_idx]}, start {start}")
        return window

