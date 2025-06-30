import torch 
from torch.utils.data import Dataset
import numpy as np
from dataset import ECGDataset

# WindowDataset class for creating overlapping windows from ECG signals (e.g. from (12, 5000) to (12, 500))
# A lead (5000 data points) is split into multiple windows (size 500 and stride 250 = 19 windows per lead)
class WindowDataset(Dataset):
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
        for file_idx in range(len(self.base)):
            for start in range(0, sample_length - window_size + 1, stride):
                self.indexes.append((file_idx, start))

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx, start = self.indexes[idx]
        sig = self.base[file_idx]  
        window = sig[start : start + self.window_size]
        if torch.isnan(window).any():
            print(f"Warning: NaN values found in window from file {self.base.files[file_idx]}, start {start}")
        if torch.isinf(window).any():
            print(f"Warning: infinite values found in window from file {self.base.files[file_idx]}, start {start}")
        return window


