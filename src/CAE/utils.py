import os
import random
import torch
import numpy as np
from dataset import ECGDataset, ECGWindowDataset

def parse_samples(root_dir):
    norm_samples = []
    abnormal_samples = []

    for folder_name in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.endswith('.hea'):
                record_id = file[:-4]
                hea_path = os.path.join(folder_path, file)
                try:
                    with open(hea_path, 'r') as f:
                        for line in f:
                            if line.startswith("# Labels:"):
                                labels = line.split(":")[1].strip().split(", ")
                                if len(labels) == 1:
                                    if labels[0] == "NORM":
                                        norm_samples.append((folder_path, record_id))
                                    else:
                                        abnormal_samples.append((folder_path, record_id))
                except Exception as e:
                    print(f"Error reading {hea_path}: {e}")
    return norm_samples, abnormal_samples


def split_datasets(root_dir, val_split=0.2, test_split=0.2, seed=42):
    random.seed(seed)
    norm, abnorm = parse_samples(root_dir)
    random.shuffle(norm)
    n_val = int(len(norm) * val_split)
    n_test_norm = int(len(norm) * test_split)

    val_norm   = norm[:n_val]
    train_norm = norm[n_val:]
    test_norm  = norm[:n_test_norm]

    random.shuffle(abnorm)
    test_abnorm = abnorm[:len(test_norm)]

    train_base = ECGDataset(train_norm)
    val_base   = ECGDataset(val_norm)
    test_base  = ECGDataset(test_norm + test_abnorm)

    return train_base, val_base, test_base

def analyze_labels(records):
    from collections import Counter
    label_counter = Counter()
    for folder_path, record_id in records:
        hea_path = os.path.join(folder_path, record_id + '.hea')
        try:
            with open(hea_path, 'r') as f:
                for line in f:
                    if line.startswith("# Labels:"):
                        labels = line.split(":")[1].strip().split(", ")
                        if len(labels) == 1:
                            label_counter[labels[0]] += 1
        except:
            continue
    return label_counter


def plot_ecg_saliency(orig_signal, saliency_signal, epoch, save_path, num_leads=12):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    lead_names = [
        "I","II","III","aVR","aVL","aVF",
        "V1","V2","V3","V4","V5","V6"
    ]
    leads_to_plot = min(num_leads, orig_signal.shape[0])
    fig_height = leads_to_plot * 2
    fig, axs = plt.subplots(leads_to_plot, 1, figsize=(12, fig_height), sharex=True)
    for ch in range(leads_to_plot):
        lead_orig = orig_signal[ch].cpu()
        axs[ch].plot(lead_orig, label="Original", alpha=0.6)
        sal = saliency_signal[ch].cpu()
        axs[ch].imshow(
            sal.unsqueeze(0),
            aspect="auto",
            extent=[0, orig_signal.shape[1], lead_orig.min(), lead_orig.max()],
            alpha=0.4,
            cmap="hot"
        )
        lead_label = lead_names[ch] if ch < len(lead_names) else f"Lead {ch}"
        axs[ch].set_title(f"Lead {lead_label} - Saliency")
    fig.suptitle(f"Epoch {epoch} - Saliency Map", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
