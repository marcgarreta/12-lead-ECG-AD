import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import torch.nn.functional as F
import numpy as np

from dataset import ECGDataset, ECGWindowDataset
from model import CAE_M
from utils import split_datasets

window_size = 500
stride = 250
num_epochs = 1000
signal_length = 5000
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reconstruct_signal(windows, window_size, stride, original_length, device="cpu"):
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


def plot_ecg_reconstruction(orig_signal, recon_signal, epoch, save_path, num_leads=12):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    lead_names = [
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6"
    ]

    leads_to_plot = min(num_leads, orig_signal.shape[0])
    fig_height = leads_to_plot * 2
    fig, axs = plt.subplots(leads_to_plot, 1, figsize=(12, fig_height), sharex=True)

    per_lead_mse = []

    for ch in range(leads_to_plot):
        lead_orig = orig_signal[ch].cpu()
        lead_recon = recon_signal[ch].cpu()
        lead_mse = F.mse_loss(lead_orig, lead_recon).item()
        per_lead_mse.append(lead_mse)

        axs[ch].plot(lead_orig, label="Original", alpha=0.6)
        axs[ch].plot(lead_recon, label="Reconstruction", alpha=0.6)
        lead_label = lead_names[ch] if ch < len(lead_names) else f"Lead {ch}"
        axs[ch].set_title(f"Lead {lead_label} - MSE: {lead_mse:.6f}")
        axs[ch].legend(loc="upper right")

    overall_mse = sum(per_lead_mse) / leads_to_plot
    fig.suptitle(f"Epoch {epoch} - Overall MSE: {overall_mse:.6f}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

def train(num_epochs=num_epochs, window_size=window_size, stride=stride, signal_length=signal_length, lr=lr, device=device):
    train_base, val_base, _ = split_datasets("/fhome/mgarreta/anomaly_detection/processed_data/")
    train_set = ECGWindowDataset(train_base, window_size=window_size, stride=stride)
    val_set = ECGWindowDataset(val_base, window_size=window_size, stride=stride)

    # Subsample for faster training
    #train_set = torch.utils.data.Subset(train_set, random.sample(range(len(train_set)), 1000))
    #val_set = torch.utils.data.Subset(val_set, random.sample(range(len(val_set)), 200))

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)

    model = CAE_M(in_channels=12).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    os.makedirs("/fhome/mgarreta/MODELS/00_CAE_M/checkpoints", exist_ok=True)
    os.makedirs("/fhome/mgarreta/MODELS/00_CAE_M/reconstructions", exist_ok=True)

    for epoch in range(num_epochs):
        # === TRAIN ===
        model.train()
        train_loss = 0.0
        progress = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}")
        for batch in progress:
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.size(0)
            progress.set_postfix(loss=loss.item())

        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}")

        # === VALIDATION ===
        model.eval()
        val_loss = 0.0
        val_progress = tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}")
        with torch.no_grad():
            for batch in val_progress:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                val_loss += loss.item() * batch.size(0)
                val_progress.set_postfix(loss=loss.item())
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}: Val Loss = {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "/fhome/mgarreta/MODELS/00_CAE_M/checkpoints/best_model_big.pth")
            print(f"✅ Saved best model at epoch {epoch+1} with val_loss = {val_loss:.6f}")

        # === VISUALIZATION ===
        with torch.no_grad():
            orig_signal = val_base[0]  # shape [5000, 12]
            if orig_signal.shape[0] == 5000 and orig_signal.shape[1] == 12:
                orig_signal = orig_signal.T  # [12, 5000]
            orig_signal = orig_signal.to(device)

            windows = [
                orig_signal[:, start:start + window_size]
                for start in range(0, signal_length - window_size + 1, stride)
                if orig_signal[:, start:start + window_size].shape[1] == window_size
            ]
            if windows:
                windows = torch.stack(windows).to(device)
                recon_windows = model(windows)
                recon_signal = reconstruct_signal(recon_windows, window_size, stride, signal_length, device)
                #mse = nn.functional.mse_loss(recon_signal, orig_signal).item()
                
                plot_ecg_reconstruction(
                    orig_signal,
                    recon_signal,
                    epoch + 1,
                    f"/fhome/mgarreta/MODELS/00_CAE_M/reconstructions/epoch_{epoch+1}_all_leads.png",
                    num_leads=5
                )
                with torch.no_grad():
                    latents = []
                    for batch in val_loader:
                        batch = batch.to(device)
                        z = model.encoder(batch)  # [B, C, T']
                        z = z.mean(dim=2)         # [B, C]
                        latents.append(z.cpu())
                    latents = torch.cat(latents, dim=0).numpy()
                    save_latent_projections(latents, epoch + 1, "/fhome/mgarreta/MODELS/00_CAE_M/latents")

def save_latent_projections(latents, epoch, save_dir):
    import umap
    from sklearn.manifold import TSNE

    os.makedirs(save_dir, exist_ok=True)

    tsne_proj_2d = TSNE(n_components=2, random_state=42).fit_transform(latents)
    tsne_proj_3d = TSNE(n_components=3, random_state=42).fit_transform(latents)
    umap_proj_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(latents)
    umap_proj_3d = umap.UMAP(n_components=3, random_state=42).fit_transform(latents)

    np.save(os.path.join(save_dir, f"epoch_{epoch}_tsne_2d.npy"), tsne_proj_2d)
    np.save(os.path.join(save_dir, f"epoch_{epoch}_tsne_3d.npy"), tsne_proj_3d)
    np.save(os.path.join(save_dir, f"epoch_{epoch}_umap_2d.npy"), umap_proj_2d)
    np.save(os.path.join(save_dir, f"epoch_{epoch}_umap_3d.npy"), umap_proj_3d)

if __name__ == "__main__":
    train()
