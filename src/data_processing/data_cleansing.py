DATA_DIR = "/fhome/mgarreta/ENTREGA/preprocessed_mimic/48481316.npy"
OUTPUT_DIR = "/fhome/mgarreta/VAE/"
# Ensure os is imported
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Load the array
arr = np.load(DATA_DIR)

# Compute NaN mask
nan_mask = np.isnan(arr)

"""# Plot NaN locations: time on x-axis, leads on y-axis
plt.figure(figsize=(12, 6))
plt.imshow(nan_mask.T, aspect='auto', interpolation='nearest', cmap='gray_r')
plt.xlabel('Time Index')
plt.ylabel('Lead Index')
plt.title('NaN Locations in ECG Array (True = NaN)')
plt.colorbar(label='NaN (1=True, 0=False)')
# Save the plot next to the data file
out_path = os.path.splitext(OUTPUT_DIR)[0] + '_nan_mask.png'
plt.savefig(out_path)
plt.close()
"""
"""# Print summary statistics
total_nans = nan_mask.sum()
per_lead = nan_mask.sum(axis=0)
per_time = nan_mask.sum(axis=1)
print(f"Total NaNs: {total_nans}")
print(f"NaN count per lead: {per_lead}")
print(f"NaN count per time index: min={per_time.min()}, max={per_time.max()}")
print(f"Saved NaN mask plot to {out_path}")
"""
# Count how many .npy files in the folder contain NaNs
data_folder = os.path.dirname(DATA_DIR)
all_files = [f for f in os.listdir(data_folder) if f.endswith('.npy')]
nan_files = []
for fname in tqdm(all_files, desc="Scanning .npy files for NaNs"):
    path = os.path.join(data_folder, fname)
    arr = np.load(path)
    if np.isnan(arr).any():
        nan_files.append(fname)
print(f"Out of {len(all_files)} files in {data_folder}, {len(nan_files)} contain NaNs.")

# Delete files containing NaNs
deleted_count = 0
for fname in nan_files:
    path = os.path.join(data_folder, fname)
    try:
        os.remove(path)
        deleted_count += 1
    except OSError as e:
        print(f"Error deleting {fname}: {e}")
# Report deletion summary
remaining_files = [f for f in os.listdir(data_folder) if f.endswith('.npy')]
print(f"Deleted {deleted_count} files containing NaNs.")
print(f"Remaining .npy files in folder: {len(remaining_files)}")
