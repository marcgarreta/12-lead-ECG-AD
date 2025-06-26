import torch
#!/usr/bin/env python3
import os
import matplotlib.gridspec as gridspec
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from captum.attr import Saliency
import copy
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from tqdm import tqdm

from model import OmniAnomalyECG
OUTPUT_DIR = "/fhome/mgarreta/ENTREGA/MA_VAE_MIMIC/OMNIANOMALY/eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR        = "/fhome/mgarreta/processed_cpsc/"
REF_CSV         = "/fhome/mgarreta/processed_reference_balanced.csv"
CHECKPOINT_PATH = "/fhome/mgarreta/ENTREGA/MA_VAE_MIMIC/OMNIANOMALY/checkpoints/VAE_test_full/best_mse_omni_ecg.pth"
WINDOW_SIZE     = 500
STRIDE          = 250
SIGNAL_LENGTH   = 5000
BATCH_SIZE      = 64
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM     = 16   

def load_cpsc_data(data_dir, ref_csv):
    df = pd.read_csv(ref_csv)
    samples = []
    for _, row in df.iterrows():
        path = os.path.join(data_dir, f"{row['Recording']}.npy")
        if os.path.exists(path):
            sig = np.load(path)            # [5000,12]
            tensor = torch.tensor(sig.T, dtype=torch.float32)  # [12,5000]
            samples.append((row['Recording'], tensor, int(row['label'])))
    return samples

def reconstruct_signal(windows, window_size, stride, original_length, device="cpu"):
    windows = windows.to(device)
    # Use a Hann window to smoothly weight overlapping segments
    hann = torch.hann_window(window_size, device=device)
    channels = windows.shape[1]
    reconstructed = torch.zeros((channels, original_length), device=device)
    counts = torch.zeros((channels, original_length), device=device)

    for i in range(windows.shape[0]):
        start = i * stride
        end = start + window_size
        if end > original_length:
            break
        # apply Hann weighting to both reconstruction and weight counts
        reconstructed[:, start:end] += windows[i] * hann.unsqueeze(0)
        counts[:, start:end]       += hann.unsqueeze(0)
    return reconstructed / counts.clamp(min=1e-8)

def create_windows(signal, window_size, stride):
    C, T = signal.shape
    windows = []
    for start in range(0, T - window_size + 1, stride):
        w = signal[:, start:start+window_size]
        if w.shape[1] == window_size:
            windows.append(w.unsqueeze(0))
    return torch.cat(windows, dim=0) if windows else torch.empty((0, C, window_size))

def evaluate_detection():
    # Load samples and model
    samples = load_cpsc_data(DATA_DIR, REF_CSV)
    #samples = samples[:]  # For testing, we can limit to first 10 samples
  # only evaluate first 10 samples
    model = OmniAnomalyECG(n_leads=12, window_size=WINDOW_SIZE, n_latent=LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # === Window-level AUC computation ===
    all_windows = torch.empty((0, 12, WINDOW_SIZE))
    all_labels = []
    for _, signal, label in samples:
        w = create_windows(signal, WINDOW_SIZE, STRIDE)  # (Nw, 12, W)
        if w.numel() == 0:
            continue
        all_windows = torch.cat([all_windows, w], dim=0)
        all_labels.extend([label] * w.shape[0])
    all_labels = np.array(all_labels)
    window_auc = evaluate_window_auc(model, all_windows, all_labels, BATCH_SIZE, DEVICE)
    print(f"Window-level AUC ROC: {window_auc:.4f}")

    y_true = []
    y_scores = []
    # For each sample, compute average window score
    for rec_id, signal, label in tqdm(samples, desc="Samples Eval"):
        # Windowing
        windows = create_windows(signal, WINDOW_SIZE, STRIDE)  # [Nw, C, W]
        if windows.numel() == 0:
            continue
        # Compute reconstructions in batches
        scores = []
        with torch.no_grad():
            for i in tqdm(range(0, len(windows), BATCH_SIZE), desc=f"Windows for {rec_id}", leave=False):
                batch = windows[i:i+BATCH_SIZE].to(DEVICE)    # (B, 12, W)
                batch = batch.permute(0, 2, 1)                # → (B, W, 12)
                x_mean, x_logvar, z, mu, logvar = model(batch)
                # Reconstruction MSE per window
                mse = ((x_mean.cpu() - batch.cpu())**2).mean(dim=[1,2])  # [B]
                scores.extend(mse.numpy().tolist())
        # Average score for this sample
        sample_score = float(np.mean(scores))
        y_true.append(label)
        y_scores.append(sample_score)

    # Determine best threshold by maximizing F1
    thresholds = np.linspace(min(y_scores), max(y_scores), 500)
    best_f1 = 0.0
    best_thr = thresholds[0]
    for thr in thresholds:
        y_pred = [1 if s > thr else 0 for s in y_scores]
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    # Final predictions
    y_pred = [1 if s > best_thr else 0 for s in y_scores]
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)

    # Print details for every sample
    print("\nSample-level details:")
    for idx, (label, score) in enumerate(zip(y_true, y_scores)):
        pred = 1 if score > best_thr else 0
        print(
            f"Sample {idx:3d}: "
            f"GT={label}, "
            f"Pred={pred}, "
            f"Score={score:.4f}, "
            f"Thr={best_thr:.4f}"
        )

    # Print results
    print(f"Anomaly detection results:")
    print(f"  Best threshold: {best_thr:.6f}")
    print(f"  F1 score: {best_f1:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  AUC ROC:   {auc:.4f}")

def evaluate_window_auc(model, windows, labels, BATCH_SIZE, DEVICE):
    model.eval()
    scores = []
    with torch.no_grad():
        for i in tqdm(range(0, len(windows), BATCH_SIZE), desc="Window AUC Eval"):
            batch = windows[i:i+BATCH_SIZE].to(DEVICE)  # (B, 12, W)
            batch = batch.permute(0, 2, 1)              # → (B, W, 12)
            x_mean, x_logvar, z, mu, logvar = model(batch)
            recon = x_mean
            mse = ((recon.cpu() - batch.cpu())**2).mean(dim=[1,2])
            scores.append(mse.numpy())
    scores = np.concatenate(scores)
    auc = roc_auc_score(labels, scores)
    return auc

if __name__ == "__main__":
    #evaluate_detection()

    model = OmniAnomalyECG(n_leads=12, window_size=WINDOW_SIZE, n_latent=LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    samples = load_cpsc_data(DATA_DIR, REF_CSV)

    num_to_plot = 3
    plotted = 0

    for rec_id, signal, label in samples:
        if label == 1:
            # Skip sample if any lead is flat
            orig_np_check = signal.cpu().numpy()
            if np.any(np.std(orig_np_check, axis=1) < 1e-6):
                continue
            # Reconstruir muestra
            windows = create_windows(signal, WINDOW_SIZE, STRIDE).to(DEVICE)       # (Nw,12,W)
            windows_p = windows.permute(0,2,1)                                    # (Nw,W,12)
            with torch.no_grad():
                x_mean, _, _, _, _ = model(windows_p)
            recon = x_mean                                                      # (Nw,W,12)
            recon_signal = reconstruct_signal(recon.permute(0,2,1), WINDOW_SIZE, STRIDE, signal.shape[1], DEVICE)
            # Convertir a numpy
            orig_np = signal.cpu().numpy()                                      # (12,T)
            recon_np = recon_signal.cpu().numpy()                               # (12,T)

            # Compute saliency map
            saliency = Saliency(lambda inp: ((model(inp)[0] - inp)**2).mean(dim=[1,2]))
            windows_p.requires_grad_()
            grads = saliency.attribute(windows_p)                               # (Nw, W, 12)
            grads_np = grads.detach().cpu().numpy().transpose(0, 2, 1)           # (Nw, 12, W)
            saliency_map = reconstruct_signal(torch.tensor(grads_np).to(DEVICE), WINDOW_SIZE, STRIDE, signal.shape[1], DEVICE)
            saliency_np = saliency_map.cpu().numpy()                            # (12, T)

            # Plot original vs reconstrucción with saliency and error
            leads, T = orig_np.shape
            # Compute reconstruction error per lead, per time step
            error = np.abs(orig_np - recon_np)
            # Create figure with GridSpec: signal plots taller than saliency and error
            fig = plt.figure(figsize=(12, 3*leads))
            # Three rows per lead: signal, saliency, error
            gs = gridspec.GridSpec(leads * 3, 1, height_ratios=[4, 1, 1] * leads)
            for i in range(leads):
                ax_signal = fig.add_subplot(gs[3*i, 0])
                ax_sal = fig.add_subplot(gs[3*i+1, 0], sharex=ax_signal)
                ax_err = fig.add_subplot(gs[3*i+2, 0], sharex=ax_signal)
                # plot original vs reconstruction
                ax_signal.plot(orig_np[i], label='Original', alpha=0.6)
                ax_signal.plot(recon_np[i], label='Reconstrucción', alpha=0.6)
                ax_signal.set_ylabel(f'L{i+1}')
                if i == 0:
                    ax_signal.legend(loc='upper right')
                # plot saliency heatmap
                ax_sal.imshow(saliency_np[i][None, :], aspect='auto', origin='lower', extent=[0, T, 0, 1], cmap='coolwarm')
                ax_sal.set_yticks([])
                # plot reconstruction error heatmap
                ax_err.imshow(error[i][None, :], aspect='auto', origin='lower', extent=[0, T, 0, 1], cmap='Reds')
                ax_err.set_yticks([])
            plt.xlabel('Tiempo')
            plt.suptitle(f'Muestra anómala {rec_id}: original vs reconstrucción')
            out_fig = os.path.join(OUTPUT_DIR, f"{rec_id}_recon_vs_orig_version_last.png")
            plt.tight_layout(rect=[0,0,0.9,1])
            plt.savefig(out_fig)
            plt.close(fig)
            print(f"Guardada visualización en {out_fig}")
            plotted += 1
            if plotted >= num_to_plot:
                break
