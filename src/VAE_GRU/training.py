import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, Subset
from torch.distributions import Normal
import wfdb
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np

from dataset import ECGDataset, WindowDataset
from model import OmniAnomalyECG

seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def plot_ecg_reconstruction(orig, recon, epoch, path, num_leads=12):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    leads_to_plot = min(num_leads, orig.shape[0])
    fig, axs = plt.subplots(leads_to_plot, 1, figsize=(12, leads_to_plot * 2), sharex=True)
    for ch in range(leads_to_plot):
        orig_ch = orig[ch].cpu() if hasattr(orig[ch], 'cpu') else orig[ch]
        recon_ch = recon[ch].cpu() if hasattr(recon[ch], 'cpu') else recon[ch]
        axs[ch].plot(orig_ch, label="Original", alpha=0.6)
        axs[ch].plot(recon_ch, label="Reconstruction", alpha=0.6)
        axs[ch].set_title(f"{lead_names[ch]} - Channel {ch}")
        axs[ch].legend()
    plt.suptitle(f"Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def loss_function(x, x_mean, x_logvar, z_mean, z_logvar, beta):
    dist = Normal(x_mean, torch.exp(0.5 * x_logvar))
    log_px = dist.log_prob(x).sum([-1,-2]).mean()   # average over B
    recon = -log_px

    kl = 0.5 * (torch.exp(z_logvar) + z_mean**2 - 1 - z_logvar)
    kl = kl.sum([-1,-2]).mean()

    return recon + beta * kl, recon, kl
        
def train(model, train_loader, val_loader, device, epochs, lr, patience, scheduler,
          base_ds=None, sample_length=None, window_size=None, stride=None, recon_dir=None):
    print(f"Starting training: epochs={epochs}, patience={patience}")
    print(f"  Train samples: {len(train_loader.dataset)}, batches: {len(train_loader)}")
    print(f"  Val   samples: {len(val_loader.dataset)}, batches: {len(val_loader)}")

    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    model.to(device)
    best_val_loss = float('inf')
    wait = 0

    best_recon_loss = float('inf')

    recon_history = []
    kl_history = []

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        beta = scheduler(epoch)

        model.train()
        train_loss = 0.0
        for batch_idx, x in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}"), 1):
            # Batch-level logging every 100 batches
            if batch_idx % 100 == 0:
                print(f"  [Epoch {epoch+1}] Training batch {batch_idx}/{len(train_loader)}")
            x = x.to(device)  # Expect (batch, seq_len, n_leads)
            optimizer.zero_grad()
            x_mean, x_logvar, z, z_mean, z_logvar = model(x)
            loss, recon_term, kl_term = loss_function(x, x_mean, x_logvar, z_mean, z_logvar, beta)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        recon_history.append(recon_term.item())
        kl_history.append((beta * kl_term).item())

        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        with torch.no_grad():
            for batch_idx, x in enumerate(tqdm(val_loader, desc=f"Val   Epoch {epoch+1}/{epochs}"), 1):
                # Validation batch logging every 50 batches
                if batch_idx % 50 == 0:
                    print(f"  [Epoch {epoch+1}] Validation batch {batch_idx}/{len(val_loader)}")
                x = x.to(device)
                x_mean, x_logvar, z, z_mean, z_logvar = model(x)
                loss, recon_term, kl_term = loss_function(x, x_mean, x_logvar, z_mean, z_logvar, beta)
                val_loss += loss.item() * x.size(0)
                val_recon_loss += recon_term.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        val_recon_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1:03d} | β={beta:.3e} | train_ELBO={train_loss:.4f} | val_ELBO={val_loss:.4f} | val_recon={val_recon_loss:.4f}")

        if val_recon_loss < best_recon_loss:
            best_recon_loss = val_recon_loss
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

if __name__ == "__main__":
    CHECK_DIR = '/fhome/mgarreta/ENTREGA/MA_VAE_MIMIC/OMNIANOMALY/checkpoints/VAE_test_FULL'
    RECON_DIR = '/fhome/mgarreta/ENTREGA/MA_VAE_MIMIC/OMNIANOMALY/recon_test_FULL'
    DATA_DIR = os.getenv("ECG_DATA_DIR", "/fhome/mgarreta/ENTREGA/preprocessed_mimic")

    os.makedirs(CHECK_DIR, exist_ok=True)
    os.makedirs(RECON_DIR, exist_ok=True)

    window_size = 250
    stride = 125
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
    #REDUCE_TRAIN = 200000
    #REDUCE_VAL = 20000
    #train_idx = random.sample(train_idx, min(REDUCE_TRAIN, len(train_idx)))
    #val_idx = random.sample(val_idx, min(REDUCE_VAL, len(val_idx)))

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
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
    model = OmniAnomalyECG(n_leads=12, n_hidden=128, n_layers=3).to(device)

    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    # Learning rate scheduler: reduce LR on plateau of validation loss
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
            )
    warmup = 20
    best_val = float('inf')
    best_val_mse = float('inf')
    train_recon_hist = []
    train_kl_hist = []
    val_recon_hist = []
    val_kl_hist = []

    for epoch in range(1, 100):
        beta = min(1.0, (epoch / warmup))
        # Cap beta so it does not exceed previous epoch's reconstruction loss
        if train_recon_hist:
            beta = min(beta, train_recon_hist[-1])

        model.train()
        tot_train = tot_recon = tot_kl = 0
        for x in tqdm(train_loader, desc=f"Train Ep{epoch}"):
            x = x.to(device)
            x_mean, x_logvar, z, mu, logvar = model(x)
            recon = F.mse_loss(x_mean, x, reduction='mean')
            # normalized KL per time-step and latent dim
            per_td_kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl = per_td_kl.mean(dim=[1,2]).mean()

            loss = recon + beta * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_train += loss.item() * x.size(0)
            tot_recon += recon.item() * x.size(0)
            tot_kl    += kl.item()    * x.size(0)

        model.eval()
        tot_val = tot_val_recon = tot_val_kl = 0
        with torch.no_grad():
            for x in tqdm(val_loader, desc=f"Val   Ep{epoch}"):
                x = x.to(device)
                x_mean_batch, x_logvar, z, mu, logvar = model(x)
                recon = F.mse_loss(x_mean_batch, x, reduction='mean')
                # normalized KL per time-step and latent dim
                per_td_kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                kl = per_td_kl.mean(dim=[1,2]).mean()
            
                tot_val       += (recon + beta * kl).item() * x.size(0)
                tot_val_recon += recon.item() * x.size(0)
                tot_val_kl    += kl.item()    * x.size(0)

        # ——— Plot de una ventana de validación ———
        # Tomar el primer batch de validación y una ventana cualquiera (ej. la ventana 0)
        x_val_batch = next(iter(val_loader)).to(device)  # (B, W, C)
        with torch.no_grad():
            x_mean_batch, x_logvar, z, mu, logvar = model(x_val_batch)
        # Seleccionar ventana 0 de la batch
        orig_win = x_val_batch[0]           # (W, C)
        recon_win = x_mean_batch[0]        # (W, C)
        # Permutar para canales x timesteps
        orig_win_plot = orig_win.permute(1, 0)   # (C, W)
        recon_win_plot = recon_win.permute(1, 0) # (C, W)
        # Guardar figura
        win_vis_path = os.path.join(RECON_DIR, f"val_win_epoch{epoch:03d}.png")
        plot_ecg_reconstruction(orig_win_plot, recon_win_plot, epoch, win_vis_path)

        train_loss = tot_train / len(train_loader.dataset)
        val_loss   = tot_val   / len(val_loader.dataset)

        mean_train_recon = tot_recon / len(train_loader.dataset)
        mean_train_kl    = tot_kl    / len(train_loader.dataset)
        mean_val_recon   = tot_val_recon / len(val_loader.dataset)
        mean_val_kl      = tot_val_kl    / len(val_loader.dataset)

        mean_train_elbo = mean_train_recon + mean_train_kl
        mean_val_elbo   = mean_val_recon + mean_val_kl
        print(f"Epoch {epoch:03d} Summary → "
              f"Train MSE={mean_train_recon:.4f}, Train KL={mean_train_kl:.4f}, Train ELBO={mean_train_elbo:.4f} | "
              f"Val MSE={mean_val_recon:.4f}, Val KL={mean_val_kl:.4f}, Val ELBO={mean_val_elbo:.4f}")

        # Save model when validation MSE improves
        if mean_val_recon < best_val_mse:
            best_val_mse = mean_val_recon
            torch.save(model.state_dict(), os.path.join(CHECK_DIR, 'best_mse_omni_ecg.pth'))
            print(f"→ Saved best-MSE model at epoch {epoch}")

        train_recon_hist.append(mean_train_recon)
        train_kl_hist.append(mean_train_kl)
        val_recon_hist.append(mean_val_recon)
        val_kl_hist.append(mean_val_kl)

        # Step the LR scheduler with the current validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(CHECK_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(CHECK_DIR, 'best_omni_ecg.pth'))
        elif epoch - list(range(epoch))[::-1][tot_val == best_val] >= 10:
            print("Early stopping")
            break
