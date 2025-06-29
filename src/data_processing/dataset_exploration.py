from pathlib import Path
from torch.utils.data import DataLoader
from dataset import ECGDataset
from window_dataset import WindowDataset
import numpy as np
import pandas as pd

# Derive processed data paths relative to project root
SCRIPT_DIR   = Path(__file__).resolve().parent       # .../src
PROJECT_ROOT = SCRIPT_DIR.parent                     # project root
DATA_PROC    = PROJECT_ROOT / 'data' / 'processed'
PTBXL_DIR    = DATA_PROC / 'ptbxl'
MIMIC_DIR    = DATA_PROC / 'mimic'

# Parameters
sample_length = 5000
window_size   = 500
stride        = 250
batch_size    = 64

# 1) Test windows on PTB-XL only
print("\n=== PTB-XL Window Dataset ===")
ds_ptb_win = WindowDataset(
    ptbxl_dir=PTBXL_DIR,
    mimic_dir=None,
    dataset='ptbxl',
    sample_length=sample_length,
    window_size=window_size,
    stride=stride,
)
print("Dataset object:", ds_ptb_win)
print("First 3 file paths:", ds_ptb_win.base.filepaths[:3])
print(f"Total windows (PTB-XL): {len(ds_ptb_win)}")
loader_ptb = DataLoader(ds_ptb_win, batch_size=batch_size, shuffle=False)
batch = next(iter(loader_ptb))
print(f"Window batch shape (PTB-XL): {batch.shape}")

# 2) Test windows on MIMIC only
print("\n=== MIMIC Window Dataset ===")
ds_mim_win = WindowDataset(
    ptbxl_dir=None,
    mimic_dir=MIMIC_DIR,
    dataset='mimic',
    sample_length=sample_length,
    window_size=window_size,
    stride=stride,
)
print("Dataset object:", ds_mim_win)
print("First 3 file paths:", ds_mim_win.base.filepaths[:3])
print(f"Total windows (MIMIC): {len(ds_mim_win)}")
loader_mim = DataLoader(ds_mim_win, batch_size=batch_size, shuffle=False)
batch = next(iter(loader_mim))
print(f"Window batch shape (MIMIC): {batch.shape}")

# 3) Test windows on both merged
print("\n=== Merged Window Dataset ===")
ds_both_win = WindowDataset(
    ptbxl_dir=PTBXL_DIR,
    mimic_dir=MIMIC_DIR,
    dataset='both',
    sample_length=sample_length,
    window_size=window_size,
    stride=stride,
)
print("Dataset object:", ds_both_win)
print("First 3 file paths:", ds_both_win.base.filepaths[:3])
print(f"Total windows (both): {len(ds_both_win)}")
loader_both = DataLoader(ds_both_win, batch_size=batch_size, shuffle=False)
batch = next(iter(loader_both))
print(f"Window batch shape (both): {batch.shape}")

# 4) Z-score verification
def check_zscore(npy_dir: Path, sample_max: int = 100) -> pd.DataFrame:
    files = list(npy_dir.glob("*.npy"))
    if len(files) > sample_max:
        files = np.random.choice(files, sample_max, replace=False)
    stats = []
    for f in files:
        arr = np.load(f)
        # transpose if (12, T)
        if arr.ndim == 2 and arr.shape[0] == 12:
            arr = arr.T
        stats.append({
            "mean": np.mean(arr),
            "std":  np.std(arr)
        })
    return pd.DataFrame(stats)

print("\n=== Z-score Verification ===")
ptb_stats = check_zscore(PTBXL_DIR)
mim_stats = check_zscore(MIMIC_DIR)
print("PTB-XL means/stds (sample):")
print(ptb_stats.describe())
print("\nMIMIC means/stds (sample):")
print(mim_stats.describe())
