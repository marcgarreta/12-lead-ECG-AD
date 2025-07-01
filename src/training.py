from pathlib import Path
import sys
# Project root is one level above src/
ROOT = Path(__file__).resolve().parents[1]
# Ensure src/ is on Python path for imports
sys.path.insert(0, str(ROOT / 'src'))
import os
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from data_processing.window_dataset import WindowDataset  
from models.vae_bilstm_attention import VAE  
from models.cae import CAE
import argparse

def cyclical_annealing_beta(epoch: int,
                            cycle_period: int = 10,
                            ramp_ratio: float = 0.5,
                            max_beta: float = 1.0) -> float:
    cycle_epoch = (epoch - 1) % cycle_period
    ramp_epochs = max(1, int(cycle_period * ramp_ratio)) 
    if cycle_epoch < ramp_epochs:
        return max_beta * (cycle_epoch + 1) / ramp_epochs
    else:
        return max_beta

def reconstruct_full_saliency(model, dataset, device):
    was_training = model.training
    if not was_training:
        model.train()
    window_size  = dataset.window_size
    sample_length = dataset.base.sample_length
    n_leads      = dataset.base[0].shape[1]
    grad_sum   = torch.zeros(sample_length, n_leads, device=device)
    grad_count = torch.zeros(sample_length, n_leads, device=device)

    for (file_idx, start) in dataset.indexes:
        window = dataset.base[file_idx][start:start+window_size].unsqueeze(0).to(device)
        window.requires_grad_(True)
        model.zero_grad(set_to_none=True)

        x_mean, _, _, _, _, _ = model(window)
        loss = torch.mean((x_mean - window) ** 2)  
        loss.backward()

        grad_abs = window.grad.abs().squeeze(0)  # (T_window, n_leads)
        grad_sum[start:start+window_size]   += grad_abs
        grad_count[start:start+window_size] += 1

        window.grad = None 

    grad_count[grad_count == 0] = 1
    saliency_full = grad_sum / grad_count
    if not was_training:
        model.eval()
    return saliency_full.cpu()

def reconstruct_full_mean(model, dataset, device):
    model.eval()
    window_size = dataset.window_size
    stride = dataset.stride
    sample_length = dataset.base.sample_length
    n_leads = dataset.base[0].shape[1]
    recon_sum = torch.zeros(sample_length, n_leads, device=device)
    recon_count = torch.zeros(sample_length, n_leads, device=device)
    with torch.no_grad():
        for idx, (file_idx, start) in enumerate(dataset.indexes):
            window = dataset.base[file_idx][start:start+window_size].unsqueeze(0).to(device)
            x_mean, _, _, _, _, _ = model(window)
            x_mean = x_mean.squeeze(0)  # (T, C)
            recon_sum[start:start+window_size] += x_mean
            recon_count[start:start+window_size] += 1
    recon_count[recon_count == 0] = 1
    recon_full = recon_sum / recon_count
    return recon_full.cpu()

def reconstruct_full_std(model, dataset, device):
    model.eval()
    window_size = dataset.window_size
    stride = dataset.stride
    sample_length = dataset.base.sample_length
    n_leads = dataset.base[0].shape[1]
    recon_sum = torch.zeros(sample_length, n_leads, device=device)
    recon_count = torch.zeros(sample_length, n_leads, device=device)
    with torch.no_grad():
        for idx, (file_idx, start) in enumerate(dataset.indexes):
            window = dataset.base[file_idx][start:start+window_size].unsqueeze(0).to(device)
            _, x_logvar, _, _, _, _ = model(window)
            sigma = torch.exp(0.5 * x_logvar.squeeze(0))  # (T, C)
            recon_sum[start:start+window_size] += sigma
            recon_count[start:start+window_size] += 1
    recon_count[recon_count == 0] = 1
    recon_full_std = recon_sum / recon_count
    return recon_full_std.cpu()

def reconstruct_full_attn(model, dataset, device):
    model.eval()
    window_size  = dataset.window_size
    stride       = dataset.stride
    sample_length = dataset.base.sample_length
    n_leads      = dataset.base[0].shape[1]
    attn_sum   = torch.zeros(sample_length, n_leads, device=device)
    attn_count = torch.zeros(sample_length, n_leads, device=device)
    with torch.no_grad():
        for (file_idx, start) in dataset.indexes:
            window = dataset.base[file_idx][start:start+window_size].unsqueeze(0).to(device)
            _, _, _, _, _, lead_w = model(window)
            lead_w = lead_w.squeeze(0)  # (T_window, n_leads)
            attn_sum[start:start+window_size]   += lead_w
            attn_count[start:start+window_size] += 1
    attn_count[attn_count == 0] = 1
    attn_full = attn_sum / attn_count
    return attn_full.cpu()
    
def train_epoch(model, dataloader, optimizer, device, args):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    for x in tqdm(dataloader, desc="Train"):
        # x: (B, T, n_leads)
        x = x.to(device)
        optimizer.zero_grad()

        if args.model == 'vae':
            x_mean, x_logvar, mu, logvar, _, _ = model(x)
            loss, recon_loss, kl_loss = model.loss_function(x, x_mean, x_logvar, mu, logvar)
        else:  # cae
            x_recon = model(x.transpose(1, 2))  # CAE expects [B, C, L]
            x = x.transpose(1, 2)
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(x_recon, x)
            recon_loss = loss
            kl_loss = 0.0

        # Backward & optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate for logging
        batch_size = x.size(0)
        total_loss  += loss.item()       * batch_size
        total_recon += recon_loss.item() * batch_size
        total_kl    += kl_loss if isinstance(kl_loss, float) else kl_loss.item() * batch_size

    # Average over all windows
    n_samples = len(dataloader.dataset)
    avg_loss  = total_loss  / n_samples
    avg_recon = total_recon / n_samples
    avg_kl    = total_kl    / n_samples

    return avg_loss, avg_recon, avg_kl

def validate_epoch(model, dataloader, device, args):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    with torch.no_grad():
        for x in tqdm(dataloader, desc="Validate"):
            x = x.to(device)
            if args.model == 'vae':
                x_mean, x_logvar, mu, logvar, _, _ = model(x)
                loss, recon_loss, kl_loss = model.loss_function(x, x_mean, x_logvar, mu, logvar)
            else:  # cae
                x_recon = model(x.transpose(1, 2))
                x = x.transpose(1, 2)
                loss_fn = torch.nn.MSELoss()
                loss = loss_fn(x_recon, x)
                recon_loss = loss
                kl_loss = 0.0

            batch_size = x.size(0)
            total_loss  += loss.item()       * batch_size
            total_recon += recon_loss.item() * batch_size
            total_kl    += kl_loss if isinstance(kl_loss, float) else kl_loss.item() * batch_size

    n_samples = len(dataloader.dataset)
    avg_loss  = total_loss  / n_samples
    avg_recon = total_recon / n_samples
    avg_kl    = total_kl    / n_samples

    return avg_loss, avg_recon, avg_kl

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, choices=['vae', 'cae'], default='vae', help='Model to train: vae or cae')
    p.add_argument('--dataset', type=str, choices=['ptbxl', 'mimic', 'both'], default='ptbxl', help='Dataset to train on: ptbxl, mimic, or both')
    args = p.parse_args()

    DATA_DIR = ROOT / "data" / "processed"
    WINDOW_SIZE  = 500
    STRIDE       = 125
    SAMPLE_LENGTH = 5000   

    num_epochs = 100

    # KL annaeling parameters
    CYCLE_PERIOD = 10   # full cycle length
    RAMP_RATIO   = 0.5  # ramp ration 
    MAX_BETA     = 0.3  # maximum beta value

    OUT_DIR_DETERMINISTIC = ROOT / "outputs" / "plot_det"
    OUT_DIR_VARIABILITY   = ROOT / "outputs" / "plot_var"
    if args.model == 'vae':
        MODEL_DIR = ROOT / "src" / "weights" / "VAE_weights"
    elif args.model == 'cae':
        MODEL_DIR = ROOT / "src" / "weights" / "CAE_weights"
    os.makedirs(str(OUT_DIR_DETERMINISTIC), exist_ok=True)
    os.makedirs(str(OUT_DIR_VARIABILITY), exist_ok=True)
    os.makedirs(str(MODEL_DIR), exist_ok=True)

    best_val_loss = float('inf')

    if args.dataset == 'ptbxl':
        full_ds = WindowDataset(
            ptbxl_dir=str(DATA_DIR / "ptbxl"),
            dataset='ptbxl',
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            sample_length=SAMPLE_LENGTH
        )
    elif args.dataset == 'mimic':
        full_ds = WindowDataset(
            mimic_dir=str(DATA_DIR / "mimic"),
            dataset='mimic',
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            sample_length=SAMPLE_LENGTH
        )
    elif args.dataset == 'both':
        full_ds = WindowDataset(
            ptbxl_dir=str(DATA_DIR / "ptbxl"),
            mimic_dir=str(DATA_DIR / "mimic"),
            dataset='both',
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            sample_length=SAMPLE_LENGTH
        )

    indices   = list(range(len(full_ds)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    # Set to None to use the whole dataset
    REDUCE_TRAIN = 20000
    REDUCE_VAL   = 2000

    if REDUCE_TRAIN is not None:
        train_idx = random.sample(train_idx, min(REDUCE_TRAIN, len(train_idx)))
    if REDUCE_VAL is not None:
        val_idx = random.sample(val_idx, min(REDUCE_VAL, len(val_idx)))

    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)

    BATCH_SIZE = 64

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,    
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model == 'vae':
        model = VAE(n_leads=12, n_latent=32).to(device)
    elif args.model == 'cae':
        model = CAE(in_channels=12).to(device)

    learning_rate = 5e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    for epoch in tqdm(range(1, num_epochs+1), desc="Epochs"):
        # Cyclical β‑annealing
        model.beta = cyclical_annealing_beta(epoch, CYCLE_PERIOD, RAMP_RATIO, MAX_BETA)

        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, device, args)
        val_loss, val_recon, val_kl = validate_epoch(model, val_loader, device, args)

        # Compute KL/recon ratios
        train_ratio = train_kl / train_recon if train_recon > 0 else float('inf')
        val_ratio   = val_kl   / val_recon   if val_recon   > 0 else float('inf')

        if args.model == 'vae':
            with torch.no_grad():
                all_mu = []
                all_logvar = []
                for x in val_loader:
                    x = x.to(device)
                    x_mean, x_logvar, mu, logvar, _, _ = model(x)
                    # mu, logvar: (B, T, D)
                    B, T, D = mu.shape
                    all_mu.append(mu.reshape(B * T, D).cpu())
                    all_logvar.append(logvar.reshape(B * T, D).cpu())
                all_mu = torch.cat(all_mu, dim=0)          
                all_logvar = torch.cat(all_logvar, dim=0) 
                post_var = all_logvar.exp().mean(dim=0)    
                mean_mu  = all_mu.mean(dim=0)              

                # Compute per-dimension KL on validation set
                kl_per = -0.5 * (1 + all_logvar - all_mu.pow(2) - all_logvar.exp())  # (N, D)
                kl_per_dim = kl_per.mean(dim=0)  # (D,)
                low_kl_dims = (kl_per_dim < 1e-3).sum().item()
                total_dims = kl_per_dim.numel()

            # Print per-dimension posterior variance and KL
            print(f"Epoch {epoch:03d} per-dim posterior variance: {post_var.tolist()}")
            print(f"Epoch {epoch:03d} per-dim posterior mean: {mean_mu.tolist()}")
            print(f"Epoch {epoch:03d} per-dim KL: {kl_per_dim.tolist()}")
        else:
            low_kl_dims = 0
            total_dims = 0
            post_var = torch.tensor(float('nan'))
            mean_mu = torch.tensor(float('nan'))

        scheduler.step(val_loss)

        # Save model for this epoch
        epoch_model_path = os.path.join(str(MODEL_DIR), f"model_epoch_{epoch:03d}.pt")
        torch.save(model.state_dict(), epoch_model_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(str(MODEL_DIR), "best_model.pt")
            torch.save(model.state_dict(), best_model_path)

        if args.model == 'vae':
            print(f"Epoch {epoch:03d} → "
                  f"Train NLL={train_recon:.4f}, Train KL={train_kl:.4f}, KL/NLL={train_ratio:.2f} | "
                  f"Val NLL={val_recon:.4f}, Val KL={val_kl:.4f}, KL/NLL={val_ratio:.2f} | "
                  f"Low-KL dims={low_kl_dims}/{total_dims} | "
                  f"Posterior Var (mean dim)={post_var.mean() if not torch.isnan(post_var).all() else float('nan'):.4f}, Posterior Mu Var={mean_mu.var() if not torch.isnan(mean_mu).all() else float('nan'):.4f} | "
                  f"LR={optimizer.param_groups[0]['lr']:.6f} | "
                  f"Beta={model.beta:.4f}")
        else:  # cae
            print(f"Epoch {epoch:03d} → " f"Train MSE={train_recon:.4f} ->" f"Val MSE={val_recon:.4f}")

if __name__ == "__main__":
    main()
