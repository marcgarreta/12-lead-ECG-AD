import numpy as np

def check_normalization(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)

    print(f"Mean: {mean:.4f}")
    print(f"Std Dev: {std:.4f}")
    print(f"Min: {min_val:.4f}")
    print(f"Max: {max_val:.4f}")

    # Check for z-score normalization
    if -0.2 < mean < 0.2 and 0.8 < std < 1.2:
        print("\nLikely Z-Score Normalized.")
    # Check for min-max normalization
    elif 0 <= min_val <= 0.1 and 0.9 <= max_val <= 1.0:
        print("\nLikely Min-Max Normalized.")
    else:
        print("\nNormalization type unclear or signal not normalized.")

if __name__ == "__main__":
    path = "/Users/marcgarreta/Documents/GitHub/12-lead-ECG-AD/data/inference_data/cpsc/processed_cpsc/A0002.npy"
    signal = np.load(path)

    print(f"Signal shape: {signal.shape}")
    check_normalization(signal)