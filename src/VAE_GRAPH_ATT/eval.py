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

from model import MA_VAE
OUTPUT_DIR = "/fhome/mgarreta/ENTREGA/MA_VAE_MIMIC/eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_sample_recon_error_attention(model, signal, WINDOW_SIZE, STRIDE, DEVICE, out_path):
    """
    signal: tensor (12, T), original ECG
    """
    # Skip entirely flat leads
    sig_np = signal.cpu().numpy()  # (12, T)
    # compute per-lead std
    stds = np.std(sig_np, axis=1)
    if np.any(stds < 1e-6):
        print("Skipping sample due to flat lead(s):", np.where(stds < 1e-6)[0])
        return False
    # 1. Window and reconstruct
    windows = create_windows(signal, WINDOW_SIZE, STRIDE).to(DEVICE)  # (Nw,12,W)
    # permute for model
    windows_p = windows.permute(0,2,1)  # (Nw, W,12)
    with torch.no_grad():
        # get mean reconstruction (deterministic)
        x_mean, _, _, *_ = model(windows_p)
    # x_mean: (Nw,W,12)
    # stitch windows back using the reconstruction mean
    recon_signal = reconstruct_signal(x_mean.permute(0,2,1), WINDOW_SIZE, STRIDE, signal.shape[1], DEVICE)
    recon_signal = recon_signal.cpu().numpy()  # (12,T)
    orig = signal.cpu().numpy()                # (12,T)

    # 2. Compute error matrix
    error = (orig - recon_signal)**2            # (12,T)

    # 3. Compute attention focus per window if available
    # If model.phase1 returns focus:
    _, attn_weights, focus = model.phase1(windows_p)
    # focus: (Nw,), stitch to full length
    focus_full = reconstruct_signal(focus.unsqueeze(1).unsqueeze(2), WINDOW_SIZE, STRIDE, signal.shape[1], DEVICE)
    focus_full = focus_full.squeeze().detach().cpu().numpy()  # (T,)

    # Plot
    T = signal.shape[1]
    t = np.arange(T)
    num_leads = orig.shape[0]
    # Setup GridSpec: 2 rows per lead (ECG + error strip), then 1 for attention
    total_rows = 2 * num_leads + 1
    height_ratios = []
    for _ in range(num_leads):
        height_ratios += [2, 0.3]
    height_ratios += [0.5]
    fig = plt.figure(figsize=(14, 2*num_leads + 4))
    gs = gridspec.GridSpec(total_rows, 1, height_ratios=height_ratios)

    # For colorbar handle
    im_err = None

    for i in range(num_leads):
        # ECG trace row
        ax = fig.add_subplot(gs[2*i, 0])
        ax.plot(t, orig[i], label='Orig', alpha=0.6)
        ax.plot(t, recon_signal[i], label='Recon', alpha=0.6)
        ax.set_ylabel(f'L{i+1}')
        if i == 0:
            ax.legend(loc='upper right')
        # Error strip row for this lead
        ax_err = fig.add_subplot(gs[2*i+1, 0], sharex=ax)
        # error[i]: (T,)
        im_err = ax_err.imshow(error[i][None, :], aspect='auto',
                               origin='lower', cmap='Reds',
                               extent=[0, T, 0, 1])
        ax_err.set_yticks([])
        ax_err.set_xticks([])
        if i < num_leads - 1:
            ax_err.set_xticklabels([])

    # Attention heatmap row
    ax_attn = fig.add_subplot(gs[total_rows-1, 0])
    im2 = ax_attn.imshow(focus_full[np.newaxis,:], aspect='auto', origin='lower',
                          extent=[0,T,0,1], cmap='viridis')
    ax_attn.set_ylabel('Attn')
    ax_attn.set_yticks([])

    # Colorbars: error (reuse last error strip axes), attention
    cax_err = fig.add_axes([0.92, 0.55, 0.015, 0.3])
    fig.colorbar(im_err, cax=cax_err, label='MSE', orientation='vertical')
    cax_att = fig.add_axes([0.92, 0.2, 0.015, 0.2])
    fig.colorbar(im2, cax=cax_att, label='Attn', orientation='vertical')

    plt.suptitle('Sample Reconstruction, Error & Attention')
    plt.tight_layout(rect=[0,0,0.9,0.96])
    plt.savefig(out_path)
    plt.close()
    return True

def evaluate_window_auc(model, windows, labels, BATCH_SIZE, DEVICE):
    model.eval()
    scores = []
    with torch.no_grad():
        for i in tqdm(range(0, len(windows), BATCH_SIZE), desc="Window AUC Eval"):
            batch = windows[i:i+BATCH_SIZE].to(DEVICE)  # (B, 12, W)
            batch = batch.permute(0, 2, 1)              # → (B, W, 12)
            out = model(batch)
            if isinstance(out, tuple):
                out = out[0]
            mse = ((out.cpu() - batch.cpu())**2).mean(dim=[1,2])
            scores.append(mse.numpy())
    scores = np.concatenate(scores)
    auc = roc_auc_score(labels, scores)
    return auc

DATA_DIR        = "/fhome/mgarreta/processed_cpsc/"
REF_CSV         = "/fhome/mgarreta/processed_reference_balanced.csv"
CHECKPOINT_PATH = "/fhome/mgarreta/ENTREGA/MA_VAE_MIMIC/check/checkpoints_localatt_small/best_ma_vae.pth"
WINDOW_SIZE     = 500
STRIDE          = 250
SIGNAL_LENGTH   = 5000
BATCH_SIZE      = 64
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM     = 64    # match the latent dimensionality used during training

# === Data Loading ===
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

# === Windowing ===
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
    samples = samples[:30]  # For testing, we can limit to first 10 samples
  # only evaluate first 10 samples
    model = MA_VAE(seq_len=WINDOW_SIZE, n_leads=12, latent_dim=LATENT_DIM).to(DEVICE)
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
                out = model(batch)
                if isinstance(out, tuple): out = out[0]
                # Reconstruction MSE per window
                mse = ((out.cpu() - batch.cpu())**2).mean(dim=[1,2])  # [B]
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


if __name__ == "__main__":
    evaluate_detection()
    samples = load_cpsc_data(DATA_DIR, REF_CSV)
    # Try each anomalous sample until a plot is made
    for rec_id, signal, label in samples:
        if label != 1:
            continue
        success = plot_sample_recon_error_attention(
            MA_VAE(seq_len=WINDOW_SIZE, n_leads=12, latent_dim=LATENT_DIM).to(DEVICE),
            signal, WINDOW_SIZE, STRIDE, DEVICE,
            os.path.join(OUTPUT_DIR, f"{rec_id}_recon.png")
        )
        if success:
            break
