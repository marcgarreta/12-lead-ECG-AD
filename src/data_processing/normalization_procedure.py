import numpy as np
import os

def z_score_normalize(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        print("WARNING: Zero standard deviation, skipping file.")
        return signal
    return (signal - mean) / std

def process_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")

                signal = np.load(file_path)
                
                # Sanity check shape (expecting [time, 12])
                if signal.ndim != 2 or signal.shape[1] != 12:
                    print(f"Skipping {file_path}, unexpected shape: {signal.shape}")
                    continue

                signal_z = z_score_normalize(signal)

                np.save(file_path, signal_z)
                print(f"Saved z-score normalized: {file_path}\n")

if __name__ == "__main__":
    folder = "/Users/marcgarreta/Documents/GitHub/12-lead-ECG-AD/data/inference_data/cpsc"
    process_folder(folder)
    print("✅ Done. All files processed.")