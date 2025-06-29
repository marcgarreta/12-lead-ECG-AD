import torch 
from torch.utils.data import Dataset
import numpy as np
from dataset import ECGDataset

class WindowDataset(Dataset):
    """
    Wraps NumpyECGDataset to yield sliding windows along the time axis.
    """
    def __init__(
        self,
        ptbxl_dir: str = None,
        mimic_dir: str = None,
        dataset: str = 'both',
        sample_length: int = 5000,
        window_size: int = 500,
        stride: int = 250,
    ):
        self.base = ECGDataset(
            ptbxl_dir=ptbxl_dir,
            mimic_dir=mimic_dir,
            dataset=dataset,
            sample_length=sample_length,
        )
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


