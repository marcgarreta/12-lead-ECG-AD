import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    """
    Loads raw ECGs saved as 2D .npy files (samples × leads) from one or two directories,
    padding or truncating each to a fixed length, and filtering by source.

    All NaN, inf, and -inf values are replaced with 0.

    Args:
        ptbxl_dir: Optional path to PTB-XL .npy files.
        mimic_dir: Optional path to MIMIC .npy files.
        dataset: 'both', 'ptbxl', or 'mimic'. Determines which sources to include.
        sample_length: Number of time samples per record; will pad or truncate to this length.
    """
    def __init__(
        self,
        ptbxl_dir: str | Path = None,
        mimic_dir: str | Path = None,
        dataset: str = 'both',
        sample_length: int = 5000,
    ):
        super().__init__()
        self.sample_length = sample_length
        self.filepaths = []

        # Validate dataset choice
        sources = []
        if dataset in ('both', 'ptbxl') and ptbxl_dir:
            sources.append(Path(ptbxl_dir))
        if dataset in ('both', 'mimic') and mimic_dir:
            sources.append(Path(mimic_dir))
        if not sources:
            raise ValueError("No valid source directories provided for ECGDataset.")

        # Gather files from chosen sources
        for src in sources:
            if not src.exists():
                raise FileNotFoundError(f"ECGDataset source not found: {src}")
            for fname in sorted(src.iterdir()):
                if fname.suffix == '.npy':
                    self.filepaths.append(fname)

        if not self.filepaths:
            raise RuntimeError("ECGDataset found no .npy files in the specified directories.")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.filepaths[idx]
        arr = np.load(str(path))  # expected shape: (samples, leads) or (leads, samples)

        # If leads first (12, T), transpose to (T, 12)
        if arr.ndim == 2 and arr.shape[0] == 12:
            arr = arr.T

        # Replace NaN and inf with 0
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if arr.ndim != 2:
            raise ValueError(f"File {path.name} has invalid shape {arr.shape}")

        sig = torch.from_numpy(arr).float()
        L, C = sig.shape  # L=time samples, C=leads (should be 12)

        # Truncate or pad to sample_length
        if L >= self.sample_length:
            sig = sig[:self.sample_length]
        else:
            pad = torch.zeros(self.sample_length - L, C, dtype=sig.dtype)
            sig = torch.cat([sig, pad], dim=0)

        return sig
