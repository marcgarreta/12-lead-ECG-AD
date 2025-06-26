import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.distributions import Normal
import wfdb
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from dataset import ECGDataset, WindowDataset
from model_spatial import MA_VAE

DATA_DIR    = os.getenv("ECG_DATA_DIR", "/fhome/mgarreta/ENTREGA/preprocessed_mimic")
RECON_DIR   = os.getenv("ECG_RECON_DIR", "/fhome/mgarreta/ENTREGA/MA_VAE_MIMIC/recons/reconstructions_localatt")
CHECK_DIR   = os.getenv("ECG_CHECK_DIR", "/fhome/mgarreta/ENTREGA/MA_VAE_MIMIC/check/checkpoints_localatt")

def loss_function(x, x_mean, x_logvar, z_mean, z_logvar, beta):
    dist = Normal(x_mean, torch.exp(0.5 * x_logvar))
    log_px = dist.log_prob(x).sum(dim=[1, 2])
    recon = -log_px.mean()
    kl = 0.5 * (torch.exp(z_logvar) + z_mean**2 - 1 - z_logvar).sum(dim=[1, 2]).mean()
    return recon + beta * kl, recon, kl

class BetaScheduler:
    def __init__(self, total_epochs, grace, start=1e-8, end=1e-2, mode='cyclical'):
        self.total = total_epochs
        self.grace = grace
        self.start = start
        self.end = end
        self.mode = mode
        self.betas = np.linspace(start, end, total_epochs)
    def __call__(self, epoch):
        if epoch < self.grace or self.mode == 'normal':
            return self.start + (self.end - self.start) * (epoch / self.grace)
        if self.mode == 'monotonic':
            return float(self.betas[min(epoch, self.total - 1)])
        idx = (epoch - self.grace) % self.total
        return float(self.betas[idx])

def reconstruct_signal(windows, window_size, stride, original_length, device="cpu"):
    channels = windows.shape[1]
    reconstructed = torch.zeros((channels, original_length), device=device)
    counts = torch.zeros((channels, original_length), device=device)

    for i in range(windows.shape[0]):
        start = i * stride
        end = start + window_size
        if end > original_length:
            break
        reconstructed[:, start:end] += windows[i]
        counts[:, start:end] += 1

    return reconstructed / counts.clamp(min=1e-8)

def plot_ecg_reconstruction(orig, recon, epoch, path, num_leads=12):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    leads_to_plot = min(num_leads, orig.shape[0])
    fig, axs = plt.subplots(leads_to_plot, 1, figsize=(12, leads_to_plot * 2), sharex=True)
    mse_list = []

    for ch in range(leads_to_plot):
        orig_ch = orig[ch].cpu()
        recon_ch = recon[ch].cpu()
        mse = F.mse_loss(orig_ch, recon_ch).item()
        mse_list.append(mse)
        axs[ch].plot(orig_ch, label="Original", alpha=0.6)
        axs[ch].plot(recon_ch, label="Reconstruction", alpha=0.6)
        axs[ch].set_title(f"{lead_names[ch] if ch < len(lead_names) else f'Lead {ch}'} - MSE: {mse:.4f}")
        axs[ch].legend()

    # Include key MA-VAE hyperparameters and move title up to avoid overlap
    #latent_dim = model.encoder.linear_mean.out_features
    #n_heads = model.ma.attn.num_heads
    #noise_std = model.encoder.noise.std
    plt.suptitle(
        f"Epoch {epoch} - MSE: {np.mean(mse_list):.4f}",
        y=1.02
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(path)
    plt.close()

def plot_full_ecg_with_attention(orig, recon, attn_leads, epoch, path):
    """
    orig: (12, L) numpy array
    recon: (12, L) numpy array
    attn_leads: (12, L) numpy array of attention scores per lead
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    num_leads, L = orig.shape
    lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    fig, axes = plt.subplots(num_leads * 2, 1, figsize=(16, num_leads*3.0), sharex=True)
    t = np.arange(L)
    # Normalize for shared colorbar
    norm = plt.Normalize(vmin=attn_leads.min(), vmax=attn_leads.max())
    for i in range(num_leads):
        ax_sig = axes[2 * i]
        ax_sig.plot(t, orig[i], label="Orig", alpha=0.6)
        ax_sig.plot(t, recon[i], label="Recon", alpha=0.6)
        ax_sig.set_ylabel(lead_names[i] if i < len(lead_names) else f"Lead {i}")
        if i == 0:
            ax_sig.legend(loc='upper right')
    # Shared attention colorbar
        # Attention row (2*i + 1)
        ax_attn = axes[2 * i + 1]
        im = ax_attn.imshow(
            attn_leads[i][np.newaxis, :],
            aspect='auto',
            cmap='viridis',
            origin='lower',
            norm=norm
        )
        ax_attn.set_ylabel("Attn")
        ax_attn.set_yticks([])
    axes[-1].set_xlabel("Sample index")
    # Shared colorbar at bottom
    cbar_ax = fig.add_axes([0.15, 0.01, 0.7, 0.02])
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', label='Attention')
    plt.suptitle(f"Epoch {epoch}: Full ECG reconstruction & attention", y=0.97)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(path)
    plt.close()

def train(model, train_loader, val_loader, device, epochs, lr, patience, scheduler,
          base_ds=None, sample_length=None, window_size=None, stride=None, recon_dir=None):
    print(f"Starting training: epochs={epochs}, patience={patience}")
    print(f"  Train samples: {len(train_loader.dataset)}, batches: {len(train_loader)}")
    print(f"  Val   samples: {len(val_loader.dataset)}, batches: {len(val_loader)}")

    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    model.to(device)
    best_val_loss = float('inf')
    wait = 0

    recon_history = []
    kl_history = []

    for epoch in range(epochs):
        # Epoch-level logging
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        beta = scheduler(epoch)

        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, x in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}"), 1):
            # Batch-level logging every 100 batches
            if batch_idx % 100 == 0:
                print(f"  [Epoch {epoch+1}] Training batch {batch_idx}/{len(train_loader)}")
            x = x.to(device)  # Expect (batch, seq_len, n_leads)
            optimizer.zero_grad()
            x_mean, x_logvar, x_recon, z_mean, z_logvar, _, _ = model(x)
            loss, recon_term, kl_term = loss_function(x, x_mean, x_logvar, z_mean, z_logvar, beta)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        recon_history.append(recon_term.item())
        kl_history.append((beta * kl_term).item())

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, x in enumerate(tqdm(val_loader, desc=f"Val   Epoch {epoch+1}/{epochs}"), 1):
                # Validation batch logging every 50 batches
                if batch_idx % 50 == 0:
                    print(f"  [Epoch {epoch+1}] Validation batch {batch_idx}/{len(val_loader)}")
                x = x.to(device)
                x_mean, x_logvar, x_recon, z_mean, z_logvar, _, _ = model(x)
                loss, _, _ = loss_function(x, x_mean, x_logvar, z_mean, z_logvar, beta)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1:03d} | β={beta:.3e} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(CHECK_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(CHECK_DIR, 'best_ma_vae.pth'))
            torch.save(model, os.path.join(CHECK_DIR, 'best_ma_vae_full.pt'))
            print(f"→ Saved full model at epoch {epoch+1}")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

        # --- Full-record visualization every 5 epochs ---
        if (epoch + 1) % 1 == 0 and base_ds is not None and sample_length is not None and window_size is not None and stride is not None and recon_dir is not None:
            # load full first record
            full_sig = base_ds.base[0].permute(1,0).cpu().numpy()  # (12, L)
            # slide windows and reconstruct
            windows = []
            attn_acc = []
            for start in range(0, sample_length - window_size + 1, stride):
                w = full_sig[:, start:start+window_size].T  # (W,12)
                w_t = torch.from_numpy(w).unsqueeze(0).to(device)  # (1,W,12)
                with torch.no_grad():
                    x_mean, x_logvar, x_recon, z_mean, z_logvar, attn_weights, _ = model(w_t)
                recon_win = x_mean.squeeze(0).permute(1,0).cpu().numpy()  # (12,W)
                # attn_weights: (1, n_leads, heads, W, W)
                aw = attn_weights[0]  # shape (n_leads, heads, W, W)
                # For each lead, average over heads and over key positions to get a vector of length W
                lead_focus = aw.mean(dim=1).mean(dim=2).cpu().numpy()  # shape (n_leads, W)
                windows.append(recon_win)
                attn_acc.append(lead_focus)
            # stitch back
            recon_full = np.zeros_like(full_sig)
            attn_full = np.zeros_like(full_sig)
            counts = np.zeros((full_sig.shape[0], full_sig.shape[1]))
            idx = 0
            for start in range(0, sample_length - window_size + 1, stride):
                end = start + window_size
                recon_full[:, start:end] += windows[idx]
                attn_full[:, start:end] += attn_acc[idx]  # each entry now shape (n_leads, W)
                counts[:, start:end] += 1
                idx += 1
            recon_full /= counts
            attn_full /= counts
            vis_path = os.path.join(recon_dir, f"full_ecg_epoch{epoch+1:03d}.png")
            plot_full_ecg_with_attention(full_sig, recon_full, attn_full, epoch+1, vis_path)
            print(f"  Saved full-ECG visualization to {vis_path}")

        # Plot recon vs beta*KL
        epochs = list(range(1, len(recon_history)+1))
        plt.figure()
        plt.plot(epochs, recon_history, label='Recon NLL')
        plt.plot(epochs, kl_history, label='β·KL')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(RECON_DIR, 'loss_components_epoch{:03d}.png'.format(epoch+1)))
        plt.close()


if __name__ == "__main__":
    os.makedirs(CHECK_DIR, exist_ok=True)
    os.makedirs(RECON_DIR, exist_ok=True)

    window_size = 500
    stride = 250
    sample_length = 5000

    dataset = WindowDataset(
        npy_folder=DATA_DIR,
        window_size=window_size,
        stride=stride,
        sample_length=sample_length
    )
    indices = list(range(len(dataset)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    # To reduce dataset size for quick testing
    REDUCE_TRAIN = 200000
    REDUCE_VAL = 20000
    train_idx = random.sample(train_idx, min(REDUCE_TRAIN, len(train_idx)))
    val_idx = random.sample(val_idx, min(REDUCE_VAL, len(val_idx)))

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,     
        num_workers=2, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,     
        num_workers=2,
        pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MA_VAE(seq_len=window_size, n_leads=12, latent_dim=64)
    scheduler = BetaScheduler(total_epochs=100, grace=10, start=1e-8, end=1e-2, mode='cyclical')

    train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=200,
        lr=1e-3,
        patience=10,
        scheduler=scheduler,
        base_ds=dataset,
        sample_length=sample_length,
        window_size=window_size,
        stride=stride,
        recon_dir=RECON_DIR
    )

    os.makedirs(CHECK_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CHECK_DIR, 'final_ma_vae_small.pth'))
    torch.save(model, os.path.join(CHECK_DIR, 'final_ma_vae_full_small.pt'))
