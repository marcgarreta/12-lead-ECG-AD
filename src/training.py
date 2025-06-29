import os
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from dataset import WindowDataset  
from model_bilstm import VAE  

def cyclical_annealing_beta(epoch: int,
                            cycle_period: int = 10,
                            ramp_ratio: float = 0.5,
                            max_beta: float = 1.0) -> float:
    """
    Compute the cyclical annealing coefficient β for the KL term of a VAE.

    A cycle consists of two phases:
    1. Linear ramp‑up phase that spans `cycle_period * ramp_ratio` epochs,
       where β increases from 0 to `max_beta`.
    2. Hold phase for the remainder of the cycle where β stays at `max_beta`.

    Parameters
    ----------
    epoch : int
        Current (1‑indexed) epoch.
    cycle_period : int
        Number of epochs in one full cycle.
    ramp_ratio : float
        Fraction of the cycle used for the linear ramp‑up (0 < ramp_ratio ≤ 1).
    max_beta : float
        Maximum value reached by β at the end of the ramp‑up.

    Returns
    -------
    float
        β value for the provided epoch.
    """
    # Epoch position inside its cycle (0‑indexed)
    cycle_epoch = (epoch - 1) % cycle_period
    ramp_epochs = max(1, int(cycle_period * ramp_ratio))  # avoid divide‑by‑zero

    if cycle_epoch < ramp_epochs:
        return max_beta * (cycle_epoch + 1) / ramp_epochs
    else:
        return max_beta

def plot_deterministic(x_orig, x_mean, epoch, lead, out_dir, attn: np.ndarray | None = None):
    os.makedirs(out_dir, exist_ok=True)
    t = range(x_orig.shape[0])
    fig, (ax_sig, ax_attn) = plt.subplots(
        2, 1, figsize=(10, 4), sharex=True, gridspec_kw={"height_ratios": [4, 1]}
    )
    ax_sig.plot(t, x_orig[:, lead], label="Original", alpha=0.7)
    ax_sig.plot(t, x_mean[:, lead], label="Reconstruction", lw=2)
    ax_sig.legend()
    ax_sig.set_title(f"Epoch {epoch}: Lead {lead} Deterministic")
    if attn is not None:
        data = np.asarray(attn)
        # ensure time dimension is the last axis
        if data.ndim == 1:
            data = data[None, :]                     # [1, T]
        elif data.shape[0] == len(t) and data.shape[1] != len(t):
            data = data.T                            # [H, T]
        # draw heat‑map
        ax_attn.imshow(data,
                       aspect="auto",
                       cmap="hot",
                       extent=[0, data.shape[-1], 0, data.shape[0]])
        ax_attn.set_yticks([])
        ax_attn.set_xticks([])
        for spine in ax_attn.spines.values():
            spine.set_visible(False)
    else:
        ax_attn.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"epoch_{epoch:03d}_lead{lead}_det.png"))
    plt.close(fig)

def plot_variability(x_orig, x_mean, x_std, epoch, lead, out_dir, attn: np.ndarray | None = None):
    os.makedirs(out_dir, exist_ok=True)
    t = range(x_orig.shape[0])
    fig, (ax_sig, ax_attn) = plt.subplots(
        2, 1, figsize=(10, 4), sharex=True, gridspec_kw={"height_ratios": [4, 1]}
    )
    ax_sig.plot(t, x_orig[:, lead], label="Original", alpha=0.7)
    ax_sig.plot(t, x_mean[:, lead], label="Reconstruction", lw=2)
    ax_sig.fill_between(
        t,
        x_mean[:, lead] - 2 * x_std[:, lead],
        x_mean[:, lead] + 2 * x_std[:, lead],
        color="gray", alpha=0.3, label="±2σ"
    )
    ax_sig.legend()
    ax_sig.set_title(f"Epoch {epoch}: Lead {lead} Variability")
    if attn is not None:
        data = np.asarray(attn)
        # ensure time dimension is the last axis
        if data.ndim == 1:
            data = data[None, :]                     # [1, T]
        elif data.shape[0] == len(t) and data.shape[1] != len(t):
            data = data.T                            # [H, T]
        # draw heat‑map
        ax_attn.imshow(data,
                       aspect="auto",
                       cmap="hot",
                       extent=[0, data.shape[-1], 0, data.shape[0]])
        ax_attn.set_yticks([])
        ax_attn.set_xticks([])
        for spine in ax_attn.spines.values():
            spine.set_visible(False)
    else:
        ax_attn.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"epoch_{epoch:03d}_lead{lead}_var.png"))
    plt.close(fig)

def plot_full_deterministic(x_orig_full, x_mean_full, epoch, lead, out_dir,
                            attn_full: np.ndarray | None = None):
    """
    Save deterministic reconstruction plot for an entire ECG lead (≈5000 time‑steps).
    If ``attn_full`` is provided ‑‑ shape (sample_length, n_leads) ‑‑ a heat‑map strip
    of the attention weights for the selected lead is drawn underneath.
    """
    os.makedirs(out_dir, exist_ok=True)
    t = range(x_orig_full.shape[0])

    # --- Two‑row layout: signal + attention strip ---
    fig, (ax_sig, ax_attn) = plt.subplots(
        2, 1, figsize=(12, 5), sharex=True, gridspec_kw={"height_ratios": [4, 1]}
    )

    # Signal
    ax_sig.plot(t, x_orig_full[:, lead], label="Original", alpha=0.7)
    ax_sig.plot(t, x_mean_full[:, lead], label="Reconstruction", lw=2)
    ax_sig.legend()
    ax_sig.set_title(f"Epoch {epoch}: Lead {lead} Full Signal Deterministic")

    # Attention heat‑map
    if attn_full is not None:
        attn_row = attn_full[:, lead].T  # (sample_length,)
        ax_attn.imshow(
            attn_row[None, :],
            aspect="auto",
            cmap="hot",
            extent=[0, len(attn_row), 0, 1]
        )
        ax_attn.set_yticks([])
        ax_attn.set_xticks([])
        for spine in ax_attn.spines.values():
            spine.set_visible(False)
    else:
        ax_attn.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"epoch_{epoch:03d}_lead{lead}_full_det.png"))
    plt.close(fig)

def plot_full_variability(x_orig_full, x_mean_full, x_std_full, epoch, lead,
                          out_dir, attn_full: np.ndarray | None = None):
    """
    Save uncertainty‑aware reconstruction plot for an entire ECG lead.
    Adds an attention heat‑map strip if ``attn_full`` is supplied.
    """
    os.makedirs(out_dir, exist_ok=True)
    t = range(x_orig_full.shape[0])

    fig, (ax_sig, ax_attn) = plt.subplots(
        2, 1, figsize=(12, 5), sharex=True, gridspec_kw={"height_ratios": [4, 1]}
    )

    # Signal + ±2σ envelope
    ax_sig.plot(t, x_orig_full[:, lead], label="Original", alpha=0.7)
    ax_sig.plot(t, x_mean_full[:, lead], label="Reconstruction", lw=2)
    ax_sig.fill_between(
        t,
        x_mean_full[:, lead] - 2 * x_std_full[:, lead],
        x_mean_full[:, lead] + 2 * x_std_full[:, lead],
        color="gray", alpha=0.3, label="±2σ"
    )
    ax_sig.legend()
    ax_sig.set_title(f"Epoch {epoch}: Lead {lead} Full Signal Variability")

    # Attention heat‑map
    if attn_full is not None:
        attn_row = attn_full[:, lead].T  # (sample_length,)
        ax_attn.imshow(
            attn_row[None, :],
            aspect="auto",
            cmap="hot",
            extent=[0, len(attn_row), 0, 1]
        )
        ax_attn.set_yticks([])
        ax_attn.set_xticks([])
        for spine in ax_attn.spines.values():
            spine.set_visible(False)
    else:
        ax_attn.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"epoch_{epoch:03d}_lead{lead}_full_var.png"))
    plt.close(fig)

    
def plot_full_multilead(x_orig_full, x_mean_full, x_std_full, epoch,
                        out_dir_det, out_dir_var,
                        attn_full: np.ndarray | None = None,
                        saliency_full: np.ndarray | None = None):
    """
    Plot all 12 leads for a full-length sample (≈5000 time‑steps) in stacked subplots,
    with a red attention heat‑map strip under each lead.

    Parameters
    ----------
    x_orig_full : (T, 12) ndarray
        Original ECG.
    x_mean_full : (T, 12) ndarray
        Reconstruction mean from the VAE.
    x_std_full  : (T, 12) ndarray
        Reconstruction σ (standard deviation).
    epoch : int
        Current epoch (for filename / title).
    out_dir_det : str
        Folder to save the deterministic figure.
    out_dir_var : str
        Folder to save the variability figure.
    attn_full : (T, 12) ndarray or None
        Lead‑wise attention weights already merged across windows.
    """
    os.makedirs(out_dir_det, exist_ok=True)
    os.makedirs(out_dir_var, exist_ok=True)

    sample_length, n_leads = x_orig_full.shape
    t = range(sample_length)

    # Guard against flat attention (vmin == vmax ⇒ imshow warning)
    if attn_full is not None:
        attn_min = float(attn_full.min())
        attn_max = float(attn_full.max())
        if abs(attn_max - attn_min) < 1e-6:
            attn_min -= 1e-6
            attn_max += 1e-6
    else:
        attn_min = attn_max = None

    # Compute min/max for MSE and saliency
    mse_full = (x_orig_full - x_mean_full) ** 2
    mse_min, mse_max = float(mse_full.min()), float(mse_full.max())
    if abs(mse_max - mse_min) < 1e-6:
        mse_min -= 1e-6; mse_max += 1e-6

    if saliency_full is not None:
        sal_min, sal_max = float(saliency_full.min()), float(saliency_full.max())
        if abs(sal_max - sal_min) < 1e-6:
            sal_min -= 1e-6; sal_max += 1e-6
    else:
        sal_min = sal_max = None

    def _plot_strip(ax, data_row, idx, cmap, vmin, vmax):
        """
        Draw a coloured strip `idx` (0=closest to signal) below the signal.
        """
        y_min, y_max = ax.get_ylim()
        strip_height = 0.15 * (y_max - y_min)
        y_top = y_min - idx * strip_height
        y_bot = y_top - strip_height
        ax.imshow(
            data_row[None, :],
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[0, sample_length, y_bot, y_top],
            zorder=0
        )
        ax.set_ylim(y_bot, y_max)

    # ---------- Deterministic ----------
    fig, axes = plt.subplots(n_leads, 1, figsize=(14, 2.2 * n_leads), sharex=True)
    for lead in range(n_leads):
        axes[lead].plot(t, x_orig_full[:, lead], label="Original", alpha=0.7)
        axes[lead].plot(t, x_mean_full[:, lead], label="Reconstruction", lw=1)
        # Plot strips: attention, MSE, saliency (if available)
        if attn_full is not None:
            _plot_strip(axes[lead], attn_full[:, lead].T, idx=0,
                        cmap="Reds", vmin=attn_min, vmax=attn_max)
        _plot_strip(axes[lead], mse_full[:, lead].T, idx=1,
                    cmap="Blues", vmin=mse_min, vmax=mse_max)
        if saliency_full is not None:
            _plot_strip(axes[lead], saliency_full[:, lead].T, idx=2,
                        cmap="Greens", vmin=sal_min, vmax=sal_max)
        axes[lead].set_ylabel(f"L{lead:02d}")
        if lead == 0:
            import matplotlib.patches as mpatches
            patches = [
                mpatches.Patch(color="red",   label="Attention"),
                mpatches.Patch(color="blue",  label="MSE"),
                mpatches.Patch(color="green", label="Saliency")
            ]
            axes[lead].legend(handles=patches, loc="upper right")
    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Epoch {epoch}: Full‑sample Deterministic (All 12 leads)")

    # ---------- Continuous legends ----------
    import matplotlib as mpl
    from matplotlib import cm

    # Normalisations
    attn_norm = mpl.colors.Normalize(vmin=attn_min, vmax=attn_max) if attn_full is not None else None
    mse_norm  = mpl.colors.Normalize(vmin=mse_min,  vmax=mse_max)
    sal_norm  = mpl.colors.Normalize(vmin=sal_min,  vmax=sal_max) if saliency_full is not None else None

    # Axes positions (left, bottom, width, height) in figure coords
    cbar_width = 0.015
    cbar_height = 0.18
    pad = 0.02
    right_edge = 0.93   # move colour bars slightly left

    if attn_full is not None:
        cax_attn = fig.add_axes([right_edge, 0.75, cbar_width, cbar_height])
        mpl.colorbar.ColorbarBase(cax_attn, cmap=cm.get_cmap("Reds"),
                                  norm=attn_norm, orientation='vertical')
        cax_attn.set_title("Attention", fontsize=8)

    cax_mse = fig.add_axes([right_edge, 0.52, cbar_width, cbar_height])
    mpl.colorbar.ColorbarBase(cax_mse, cmap=cm.get_cmap("Blues"),
                              norm=mse_norm, orientation='vertical')
    cax_mse.set_title("MSE", fontsize=8)

    if saliency_full is not None:
        cax_sal = fig.add_axes([right_edge, 0.29, cbar_width, cbar_height])
        mpl.colorbar.ColorbarBase(cax_sal, cmap=cm.get_cmap("Greens"),
                                  norm=sal_norm, orientation='vertical')
        cax_sal.set_title("Saliency", fontsize=8)

    # Slight tight‑layout again to avoid overlap with new axes
    fig.tight_layout(rect=[0, 0, 0.91, 0.97])
    fig.savefig(os.path.join(out_dir_det, f"epoch_{epoch:03d}_full_multilead_det.png"))
    plt.close(fig)

    # ---------- Variability ----------
    fig, axes = plt.subplots(n_leads, 1, figsize=(14, 2.2 * n_leads), sharex=True)
    for lead in range(n_leads):
        axes[lead].plot(t, x_orig_full[:, lead], label="Original", alpha=0.7)
        axes[lead].plot(t, x_mean_full[:, lead], label="Reconstruction", lw=1)
        axes[lead].fill_between(
            t,
            x_mean_full[:, lead] - 2 * x_std_full[:, lead],
            x_mean_full[:, lead] + 2 * x_std_full[:, lead],
            color="gray", alpha=0.3
        )
        # Plot strips: attention, MSE, saliency (if available)
        if attn_full is not None:
            _plot_strip(axes[lead], attn_full[:, lead].T, idx=0,
                        cmap="Reds", vmin=attn_min, vmax=attn_max)
        _plot_strip(axes[lead], mse_full[:, lead].T, idx=1,
                    cmap="Blues", vmin=mse_min, vmax=mse_max)
        if saliency_full is not None:
            _plot_strip(axes[lead], saliency_full[:, lead].T, idx=2,
                        cmap="Greens", vmin=sal_min, vmax=sal_max)
        axes[lead].set_ylabel(f"L{lead:02d}")
        # Optionally: add legend only to deterministic plot to avoid repetition
    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Epoch {epoch}: Full‑sample Variability (All 12 leads)")

    # Re‑use the same norms
    if attn_full is not None:
        cax_attn_var = fig.add_axes([right_edge, 0.75, cbar_width, cbar_height])
        mpl.colorbar.ColorbarBase(cax_attn_var, cmap=cm.get_cmap("Reds"),
                                  norm=attn_norm, orientation='vertical')
        cax_attn_var.set_title("Attention", fontsize=8)

    cax_mse_var = fig.add_axes([right_edge, 0.52, cbar_width, cbar_height])
    mpl.colorbar.ColorbarBase(cax_mse_var, cmap=cm.get_cmap("Blues"),
                              norm=mse_norm, orientation='vertical')
    cax_mse_var.set_title("MSE", fontsize=8)

    if saliency_full is not None:
        cax_sal_var = fig.add_axes([right_edge, 0.29, cbar_width, cbar_height])
        mpl.colorbar.ColorbarBase(cax_sal_var, cmap=cm.get_cmap("Greens"),
                                  norm=sal_norm, orientation='vertical')
        cax_sal_var.set_title("Saliency", fontsize=8)

    fig.tight_layout(rect=[0, 0, 0.91, 0.97])
    fig.savefig(os.path.join(out_dir_var, f"epoch_{epoch:03d}_full_multilead_var.png"))
    plt.close(fig)


def reconstruct_full_saliency(model, dataset, device):
    """
    Estimate gradient‑based saliency |∂MSE/∂x| for a full‑length sample.
    Returns tensor of shape (sample_length, n_leads) with absolute gradients.
    """
    # cuDNN LSTM/GRU backward fails in eval mode; temporarily switch to train()
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
        loss = torch.mean((x_mean - window) ** 2)  # scalar
        loss.backward()

        grad_abs = window.grad.abs().squeeze(0)  # (T_window, n_leads)
        grad_sum[start:start+window_size]   += grad_abs
        grad_count[start:start+window_size] += 1

        window.grad = None  # free memory

    grad_count[grad_count == 0] = 1
    saliency_full = grad_sum / grad_count
    # Restore original mode
    if not was_training:
        model.eval()
    return saliency_full.cpu()

def reconstruct_full_mean(model, dataset, device):
    """
    Reconstruct full-length ECG sample mean by merging window outputs.
    """
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
    """
    Reconstruct full-length ECG sample standard deviation by merging window sigma outputs.
    """
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

# Inserted function: reconstruct_full_attn
def reconstruct_full_attn(model, dataset, device):
    """
    Reconstruct (merge) lead‑wise attention weights for a full‑length ECG sample.
    Returns tensor of shape (sample_length, n_leads).
    """
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
            # model returns: x_mean, x_logvar, mu, logvar, A_weights, lead_w
            _, _, _, _, _, lead_w = model(window)
            lead_w = lead_w.squeeze(0)  # (T_window, n_leads)
            attn_sum[start:start+window_size]   += lead_w
            attn_count[start:start+window_size] += 1
    attn_count[attn_count == 0] = 1
    attn_full = attn_sum / attn_count
    return attn_full.cpu()
    
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    for x in tqdm(dataloader, desc="Train"):
        # x: (B, T, n_leads)
        x = x.to(device)
        optimizer.zero_grad()

        # Forward pass through our vectorized VAE
        x_mean, x_logvar, mu, logvar, _, _ = model(x)

        # Compute loss (MSE + beta * KL)
        loss, recon_loss, kl_loss = model.loss_function(x, x_mean, x_logvar, mu, logvar)

        # Backward & optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate for logging
        batch_size = x.size(0)
        total_loss  += loss.item()       * batch_size
        total_recon += recon_loss.item() * batch_size
        total_kl    += kl_loss.item()    * batch_size

    # Average over all windows
    n_samples = len(dataloader.dataset)
    avg_loss  = total_loss  / n_samples
    avg_recon = total_recon / n_samples
    avg_kl    = total_kl    / n_samples

    return avg_loss, avg_recon, avg_kl

def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    with torch.no_grad():
        for x in tqdm(dataloader, desc="Validate"):
            x = x.to(device)
            x_mean, x_logvar, mu, logvar, _, _ = model(x)
            loss, recon_loss, kl_loss = model.loss_function(x, x_mean, x_logvar, mu, logvar)

            batch_size = x.size(0)
            total_loss  += loss.item()       * batch_size
            total_recon += recon_loss.item() * batch_size
            total_kl    += kl_loss.item()    * batch_size

    n_samples = len(dataloader.dataset)
    avg_loss  = total_loss  / n_samples
    avg_recon = total_recon / n_samples
    avg_kl    = total_kl    / n_samples

    return avg_loss, avg_recon, avg_kl

def main():
    DATA_DIR = "/fhome/mgarreta/ENTREGA/preprocessed_mimic"
    WINDOW_SIZE  = 500
    STRIDE       = 125
    SAMPLE_LENGTH = 5000   


    num_epochs = 100

    CYCLE_PERIOD = 10   # epochs that form one full cycle
    RAMP_RATIO   = 0.5  # portion (0‑1) of each cycle used for a linear ramp‑up
    MAX_BETA     = 0.3  # upper bound for β during training

    OUT_DIR_DETERMINISTIC = "/fhome/mgarreta/VAE/plot/VAE_BiLSTM_CyclicalAnn_ATT_32latentBIG/plot_det"
    OUT_DIR_VARIABILITY   = "/fhome/mgarreta/VAE/plot/VAE_BiLSTM_CyclicalAnn_ATT_32latentBIG/plot_var"
    MODEL_DIR = "/fhome/mgarreta/VAE/models/VAE_BiLSTM_CyclicalAnn_ATT_32latentBIG"
    os.makedirs(OUT_DIR_DETERMINISTIC, exist_ok=True)
    os.makedirs(OUT_DIR_VARIABILITY, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    best_val_loss = float('inf')

    full_ds = WindowDataset(
        npy_folder=DATA_DIR,
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
    REDUCE_TRAIN = 1000000
    REDUCE_VAL   = 5000

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
    model = VAE(n_leads=12, n_latent=32).to(device)
    # Adjusted learning rate
    learning_rate = 5e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    for epoch in tqdm(range(1, num_epochs+1), desc="Epochs"):
        # Cyclical β‑annealing
        model.beta = cyclical_annealing_beta(epoch, CYCLE_PERIOD, RAMP_RATIO, MAX_BETA)

        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_recon, val_kl = validate_epoch(model, val_loader, device)

        # Compute KL/recon ratios
        train_ratio = train_kl / train_recon if train_recon > 0 else float('inf')
        val_ratio   = val_kl   / val_recon   if val_recon   > 0 else float('inf')

        # Inspect posterior statistics on validation data
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
            all_mu = torch.cat(all_mu, dim=0)           # (N, D)
            all_logvar = torch.cat(all_logvar, dim=0)   # (N, D)
            post_var = all_logvar.exp().mean(dim=0)     # average posterior variance per dim
            mean_mu  = all_mu.mean(dim=0)               # average posterior mean per dim

            # Compute per-dimension KL on validation set
            kl_per = -0.5 * (1 + all_logvar - all_mu.pow(2) - all_logvar.exp())  # (N, D)
            kl_per_dim = kl_per.mean(dim=0)  # (D,)
            low_kl_dims = (kl_per_dim < 1e-3).sum().item()
            total_dims = kl_per_dim.numel()

        # Print per-dimension posterior variance and KL
        print(f"Epoch {epoch:03d} per-dim posterior variance: {post_var.tolist()}")
        print(f"Epoch {epoch:03d} per-dim posterior mean: {mean_mu.tolist()}")
        print(f"Epoch {epoch:03d} per-dim KL: {kl_per_dim.tolist()}")

        scheduler.step(val_loss)

        # Save model for this epoch
        epoch_model_path = os.path.join(MODEL_DIR, f"model_epoch_{epoch:03d}.pt")
        torch.save(model.state_dict(), epoch_model_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(MODEL_DIR, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch {epoch:03d} → "
              f"Train NLL={train_recon:.4f}, Train KL={train_kl:.4f}, KL/NLL={train_ratio:.2f} | "
              f"Val NLL={val_recon:.4f}, Val KL={val_kl:.4f}, KL/NLL={val_ratio:.2f} | "
              f"Low-KL dims={low_kl_dims}/{total_dims} | "
              f"Posterior Var (mean dim)={post_var.mean():.4f}, Posterior Mu Var={mean_mu.var():.4f} | "
              f"LR={optimizer.param_groups[0]['lr']:.6f} | "
              f"Beta={model.beta:.4f}")

        # Plot example reconstructions after validation
        # Grab one batch from validation
        x_batch = next(iter(val_loader)).cpu()  # shape (B, T, n_leads)
        x_val = x_batch[0]  # first window
        # Run through model
        x_mean, x_logvar, _, _, _, _ = model(x_batch.to(device))
        x_mean = x_mean[0].detach().cpu().numpy()  # (T, n_leads)
        x_std  = torch.exp(0.5 * x_logvar[0]).detach().cpu().numpy()  # (T, n_leads)
        # Plot lead 0 by default
        lead_idx = 0
        # Compute attention weights for visualization
        x_mean_, x_logvar_, _, _, _, attn_weights = model(x_batch.to(device))
        attn = attn_weights[0].detach().cpu().numpy()  # (T, n_leads)
        plot_deterministic(
            x_val.detach().numpy(), x_mean, epoch, lead_idx, OUT_DIR_DETERMINISTIC, attn[:, lead_idx]
        )
        plot_variability(
            x_val.detach().numpy(), x_mean, x_std, epoch, lead_idx, OUT_DIR_VARIABILITY, attn[:, lead_idx]
        )

        # Reconstruct and save full sample from first file
        first_file_ds = WindowDataset(
            npy_folder=DATA_DIR,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            sample_length=SAMPLE_LENGTH
        )
        # Filter to only windows from the first file (file_idx = 0)
        first_file_ds.indexes = [(file_idx, start) for (file_idx, start) in first_file_ds.indexes if file_idx == 0]
        recon_full = reconstruct_full_mean(model, first_file_ds, device)  # (5000, 12)
        out_path = os.path.join(OUT_DIR_DETERMINISTIC, "full_reconstruction.npy")
        np.save(out_path, recon_full.numpy())

        # Reconstruct full attention map
        attn_full = reconstruct_full_attn(model, first_file_ds, device)  # (5000, 12)
        np.save(os.path.join(OUT_DIR_DETERMINISTIC, "full_attention.npy"), attn_full.numpy())

        # Reconstruct saliency map
        saliency_full = reconstruct_full_saliency(model, first_file_ds, device)  # (5000,12)
        np.save(os.path.join(OUT_DIR_DETERMINISTIC, "full_saliency.npy"), saliency_full.numpy())

        # Also reconstruct and save full sigma map
        recon_full_std = reconstruct_full_std(model, first_file_ds, device)  # (5000, 12)
        out_std = os.path.join(OUT_DIR_VARIABILITY, "full_reconstruction_std.npy")
        np.save(out_std, recon_full_std.numpy())

        # ----- Multi‑lead visualisation -----
        orig_full = first_file_ds.base[0].numpy()  # (5000, 12)
        # Plot each lead with its own attention strip in stacked subplots
        plot_full_multilead(
            orig_full,
            recon_full.numpy(),
            recon_full_std.numpy(),
            epoch,
            OUT_DIR_DETERMINISTIC,
            OUT_DIR_VARIABILITY,
            attn_full.numpy(),
            saliency_full.numpy()
        )
        
if __name__ == "__main__":
    main()
