import sys
from pathlib import Path
# ensure project root on path so `src` can be imported
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os, math, json, argparse, numpy as np, pandas as pd, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from tqdm.auto import tqdm
import torch

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from src.models.vae_bilstm_attention import VAE

from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
IMG_DIR = PROJECT_ROOT / 'img'
PLOTS_VAE_DIR = IMG_DIR / 'plots_vae'

BASE_PLOT_DIR_VAE = str(PLOTS_VAE_DIR)
WINDOW = 500
STRIDE = 250

BETA   = 0.3                   
ALPHA  = 0.7          
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

WINDOW = 500
STRIDE = 250
ATTN_WINDOW = 100    
ATTN_STRIDE = 50     
BETA   = 0.3                   
ALPHA  = 0.7          
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def slide_windows(sig):
    import torch
    if not isinstance(sig, torch.Tensor):
        sig = torch.tensor(sig, dtype=torch.float32)
    if sig.ndim != 2:
        raise ValueError(f"Expected 2D signal, got {sig.ndim}D")
    if sig.shape[0] == 12:
        sig_arr = sig
    elif sig.shape[1] == 12:
        sig_arr = sig.T
    else:
        if sig.shape[0] > 12:
            sig_arr = sig[:12, :]
        elif sig.shape[1] > 12:
            sig_arr = sig[:, :12].T
        else:
            raise ValueError(f"Unexpected signal shape {sig.shape}, expected one dimension = 12")
    T = sig_arr.shape[1]
    if T < WINDOW:
        sig_arr = torch.nn.functional.pad(sig_arr, (0, WINDOW - T))
        T = WINDOW
    # Transpose to [T, 12] for windowing
    sig_T = sig_arr.T  
    wins = []
    for start in range(0, T - WINDOW + 1, STRIDE):
        wins.append(sig_T[start:start+WINDOW])
    return torch.stack(wins)  # [n_windows, WINDOW, 12]

def reconstruct_full(model, sig):
    win_tensor = slide_windows(sig).to(DEVICE)          # [19,500,12]
    with torch.no_grad():
        x_mean, _, _, _, _ = model.forward(win_tensor)  
    recon = torch.zeros((5000, 12), device=DEVICE)
    counts = torch.zeros((5000, 12), device=DEVICE)
    for i, start in enumerate(range(0, 5000 - WINDOW + 1, STRIDE)):
        recon[start:start+WINDOW] += x_mean[i]
        counts[start:start+WINDOW] += 1
    counts[counts == 0] = 1
    recon = recon / counts

def reconstruct_full_mean_std(model, sig):
    win_tensor = slide_windows(sig).to(DEVICE)   # [19,500,12]
    with torch.no_grad():
        x_mean, x_logvar, *_ = model.forward(win_tensor)
    x_std = torch.exp(0.5 * x_logvar)            # [19,500,12]

    recon_m = torch.zeros((5000, 12), device=DEVICE)
    recon_s = torch.zeros((5000, 12), device=DEVICE)
    counts   = torch.zeros((5000, 12), device=DEVICE)

    for i, start in enumerate(range(0, 5000 - WINDOW + 1, STRIDE)):
        recon_m[start:start+WINDOW] += x_mean[i]
        recon_s[start:start+WINDOW] += x_std[i]
        counts  [start:start+WINDOW] += 1

    counts[counts == 0] = 1
    recon_m = (recon_m / counts).T.cpu()   # [12,5000]
    recon_s = (recon_s / counts).T.cpu()
    return recon_m, recon_s

def ecg_score(model, sig):
    win_tensor = slide_windows(sig).to(DEVICE)          # [19,500,12]
    x_mean, x_logvar, mu, logvar, attn, _ = model.forward(win_tensor)

    sigma2 = torch.exp(x_logvar)                # es la varianza
    err = (win_tensor - x_mean) ** 2 / (sigma2 + 1e-6)   # error normalizado

    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=[1, 2])  # [19]
    var_term = x_logvar                                  # log σ²
    base = ALPHA * err.sum(dim=[1, 2]) + (1 - ALPHA) * var_term.sum(dim=[1, 2]) + BETA * kl

    # Attention
    if attn.dim() == 4:                                  # (B, heads, T, T)
        alpha = attn.mean(dim=(1, 2, 3))                 # [19]
    else:                                                # (B, T, T)
        alpha = attn.mean(dim=(1, 2))                    # [19]

    scores = base                                # [19]
    return float(scores.mean())

def find_best_threshold(y_true, y_scores):
    best_f1, best_thr = -1, 0
    for thr in np.percentile(y_scores, np.linspace(0, 100, 100)):
        y_hat = (y_scores > thr).astype(int)
        f1 = f1_score(y_true, y_hat)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1

def plot_full_multilead(x_orig_full, x_mean_full, x_std_full, epoch,
                        out_dir_det, out_dir_var,
                        attn_full: np.ndarray | None = None,
                        anomaly_full: np.ndarray | None = None,
                        true_lbl: int | None = None,
                        pred_lbl: int | None = None):

    os.makedirs(out_dir_det, exist_ok=True)
    T, n_leads = x_orig_full.shape
    t = np.arange(T)
    subplot_titles = sum([
        [f"Lead {i+1}", "", "", ""]
        for i in range(n_leads)
    ], [])
    if attn_full is not None:
        attn_min = float(np.nanmin(attn_full))
        attn_max = float(np.nanmax(attn_full))
        if abs(attn_max - attn_min) < 1e-6:
            attn_min -= 1e-6
            attn_max += 1e-6
        attn_p1 = np.percentile(attn_full, 1)
        attn_p99 = np.percentile(attn_full, 99)
    else:
        attn_min = attn_max = None
        attn_p1 = attn_p99 = None

    mse_full = (x_orig_full - x_mean_full) ** 2
    mse_min = float(mse_full.min())
    mse_max = float(mse_full.max())
    if abs(mse_max - mse_min) < 1e-6:
        mse_min -= 1e-6
        mse_max += 1e-6

    if anomaly_full is not None:
        sal_min = float(np.nanmin(anomaly_full))
        sal_max = float(np.nanmax(anomaly_full))
        if not np.isfinite(sal_min) or not np.isfinite(sal_max):
            sal_min, sal_max = 0.0, 0.0
        elif abs(sal_max - sal_min) < 1e-6:
            sal_min -= 1e-6
            sal_max += 1e-6
    else:
        sal_min = sal_max = None

    fig = plt.figure(figsize=(14, 6 * n_leads), constrained_layout=True)
    gs = gridspec.GridSpec(n_leads * 4, 1,
                           height_ratios=[8,8,2,2] * n_leads,
                           hspace=0.2)

    def _plot_strip(ax, data, idx, cmap, vmin=None, vmax=None):
        y_min, y_max = ax.get_ylim()
        total_h = y_max - y_min
        strip_height = 0.15 * total_h
        gap = 0.2 * total_h
        # position strips with spacing
        y_top = y_min - idx * (strip_height + gap)
        y_bot = y_top - strip_height
        extent = [t[0], t[-1], y_bot, y_top]
        ax.imshow(data[None, :], aspect='auto', cmap=cmap,
                  vmin=vmin, vmax=vmax,
                  extent=extent)

    for i in range(n_leads):
        # 1) Original & Reconstruction
        ax0 = fig.add_subplot(gs[4*i])
        ax0.plot(t, x_orig_full[:, i], color='black', lw=1)
        ax0.plot(t, x_mean_full[:, i], color='C1', lw=1, ls='--')
        ax0.fill_between(t,
                         x_mean_full[:, i] - 2*x_std_full[:, i],
                         x_mean_full[:, i] + 2*x_std_full[:, i],
                         color='C1', alpha=0.3)
        # Per-lead y-axis scaling
        lead_min = float(np.nanmin(x_orig_full[:, i]))
        lead_max = float(np.nanmax(x_orig_full[:, i]))
        rec_min = float(np.nanmin(x_mean_full[:, i] - 2*x_std_full[:, i]))
        rec_max = float(np.nanmax(x_mean_full[:, i] + 2*x_std_full[:, i]))
        ymin = min(lead_min, rec_min)
        ymax = max(lead_max, rec_max)
        if not np.isfinite(ymin) or not np.isfinite(ymax) or abs(ymax - ymin) < 1e-6:
            ymin, ymax = -1, 1
        ax0.set_ylim(ymin, ymax)
        ax0.set_ylabel(f'Lead {i+1}', labelpad=8)
        if i == 0:
            ax0.legend(['Original','Recon'], loc='upper right', fontsize='small')

        # 2) Anomaly score line (same height as ax0)
        ax1 = fig.add_subplot(gs[4*i+1], sharex=ax0)
        if anomaly_full is not None:
            lead_anom = anomaly_full[:, i]
            ax1.plot(t, lead_anom, color='green', lw=1)
            fin = np.isfinite(lead_anom)
            if fin.any():
                amin = float(np.nanmin(lead_anom[fin]))
                amax = float(np.nanmax(lead_anom[fin]))
                if abs(amax - amin) < 1e-6:
                    amax = amin + 1e-6
                ax1.set_ylim(amin, amax)
            else:
                ax1.set_ylim(0.0, 1.0)
            ax1.set_ylabel('AS')
            if i < n_leads-1:
                ax1.set_xticks([])
        else:
            ax1.axis('off')

        # 3) Attention
        ax2 = fig.add_subplot(gs[4*i+2], sharex=ax0)
        if attn_full is not None:
            raw_att = attn_full[:, i]
            data_att = gaussian_filter1d(raw_att, sigma=2)
            ax2.imshow(
                data_att[None, :], aspect='auto', cmap='viridis',
                vmin=attn_p1, vmax=attn_p99,
                extent=[t[0], t[-1], ax2.get_ylim()[0], ax2.get_ylim()[1]],
                interpolation='bilinear', zorder=0
            )
        else:
            ax2.axis('off')
        if i < n_leads-1:
            ax2.set_xticks([])
        ax2.set_yticks([])

        # 4) MSE
        ax3 = fig.add_subplot(gs[4*i+3], sharex=ax0)
        mse = (x_orig_full[:, i] - x_mean_full[:, i])**2
        mse_vmax = np.percentile(mse, 99)
        _plot_strip(ax3, mse, idx=0, cmap='Blues',
                    vmin=0, vmax=mse_vmax)
        if i < n_leads-1:
            ax3.set_xticks([])
        ax3.set_yticks([])


    fig.get_axes()[-1].set_xlabel('Time')
    plt.suptitle(f'Epoch {epoch}: true={true_lbl}, pred={pred_lbl}', fontsize=14, y=0.88)

    import matplotlib as mpl
    from matplotlib import cm
    # Use percentile-based attn_p1/attn_p99 for colorbar normalization
    if attn_full is not None:
        attn_norm = mpl.colors.Normalize(vmin=attn_p1, vmax=attn_p99)
    else:
        attn_norm = None
    mse_norm = mpl.colors.Normalize(vmin=mse_min, vmax=mse_max)

    cbar_width = 0.015
    right_edge = 0.93
    # Attention colorbar
    if attn_full is not None:
        cax_attn = fig.add_axes([right_edge, 0.75, cbar_width, 0.10])
        cbar_attn = mpl.colorbar.ColorbarBase(cax_attn, cmap=mpl.colormaps['viridis'], norm=attn_norm, orientation='vertical')
        cbar_attn.ax.yaxis.offsetText.set_visible(False)
        cax_attn.set_title('Attention', fontsize=8)
        cax_attn.tick_params(labelsize=8)
    # MSE colorbar
    cax_mse = fig.add_axes([right_edge, 0.55, cbar_width, 0.10])
    cbar_mse = mpl.colorbar.ColorbarBase(cax_mse, cmap=mpl.colormaps['Blues'], norm=mse_norm, orientation='vertical')
    cbar_mse.ax.yaxis.offsetText.set_visible(False)
    cax_mse.set_title('MSE', fontsize=8)
    cax_mse.tick_params(labelsize=6)
    out_path = os.path.join(out_dir_det, f'epoch_{epoch:03d}_full_multilead.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'➜ Guardado {out_path}')

@torch.no_grad()
def main(args):
    import glob, os, pandas as pd, random
    samples = None
    if args.cpsc_dir and args.cpsc_csv:
        cpsc_df = pd.read_csv(args.cpsc_csv)
        cpsc_df['path'] = cpsc_df['Recording'].astype(str).apply(
            lambda r: os.path.join(args.cpsc_dir, f"{r}.npy"))
        initial_count = len(cpsc_df)
        cpsc_df = cpsc_df[cpsc_df.path.map(os.path.exists)]
        valid_count = len(cpsc_df)

        samples = [(row['path'], int(row['label']), row['Recording'])
                   for _, row in cpsc_df.iterrows()]
        if args.cpsc_max_samples is not None:
            samples = random.sample(samples, min(args.cpsc_max_samples, len(samples)))
    else:
        raise ValueError("Por favor especifique un directorio de visualización y su CSV")

    # 1. modelo
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)

    if isinstance(ckpt, dict):
        model = VAE(n_leads=12, n_latent=64).to(DEVICE)  
        model.load_state_dict(ckpt)
    else:                           
        model = ckpt.to(DEVICE)

    model.eval()
    examples = []  

    print()
    ids, ys, scores = [], [], []
    for idx_ex, (fp, true_lbl, rid_ex) in enumerate(samples, start=1):
        sig = torch.tensor(np.load(fp), dtype=torch.float32)
        if sig.ndim == 2 and sig.shape[0] != 12 and sig.shape[1] == 12:
            sig = sig.T
        elif sig.ndim == 2 and sig.shape[0] > 12 and sig.shape[1] > 12:
            sig = sig[:12, :]
        elif sig.ndim != 2:
            raise ValueError(f"Expected 2D signal, got {sig.ndim}D")
        L = sig.shape[1]
        if L < 5000:
            sig = F.pad(sig, (0, 5000 - L))
        elif L > 5000:
            sig = sig[:, :5000]

        examples.append({'sig': sig, 'rid': rid_ex, 'true': int(true_lbl)})

        ids.append(rid_ex)
        ys.append(int(true_lbl))
        scores.append(ecg_score(model, sig))

    ys = np.array(ys)
    scores = np.array(scores)

    thr, _ = find_best_threshold(ys, scores)

    out_dir_full = os.path.join(args.plot_dir, 'full_lead')
    out_dir_windows = os.path.join(args.plot_dir, 'windows')
    os.makedirs(out_dir_full, exist_ok=True)
    os.makedirs(out_dir_windows, exist_ok=True)

    for idx_ex, ex in enumerate(examples, start=1):
        sig_ex  = ex['sig']
        rid_ex  = ex['rid']
        true_lbl = ex['true']

        # Compute BPM (use of lead II for R-peak detection)
        ecg_ch = sig_ex[1].cpu().numpy()
        peaks, _ = find_peaks(ecg_ch, distance=0.4 * 500)
        if len(peaks) >= 2:
            times = peaks / 500.0
            rr = np.diff(times)
            bpm_inst = 60.0 / rr   #
            bpm_mean = float(np.mean(bpm_inst))
            unstable = np.where(np.abs(bpm_inst - bpm_mean) > 10)[0]
        else:
            bpm_mean = float('nan')
            unstable = np.array([], dtype=int)

        unstable_times = (peaks[unstable+1] if len(unstable)>0 else np.array([],dtype=int))

        T = sig_ex.shape[1]
        n_leads = sig_ex.shape[0]
        attn_full = np.zeros((T, n_leads))
        for lead in range(n_leads):
            sig_1ch = torch.zeros_like(sig_ex)
            sig_1ch[lead] = sig_ex[lead]
            win_tensor = []
            sig_arr = sig_1ch
            for start in range(0, T - ATTN_WINDOW + 1, ATTN_STRIDE):
                win_tensor.append(sig_arr[:, start:start+ATTN_WINDOW].T)
            win_1ch = torch.stack(win_tensor).to(DEVICE)  # [n_blocks, ATTN_WINDOW, 12]
            with torch.no_grad():
                *_, attn_1ch = model.forward(win_1ch)
            if attn_1ch.ndim == 4:
                a = attn_1ch.mean(dim=(1,2,3)).cpu().numpy()
            else:
                a = attn_1ch.mean(dim=(1,2)).cpu().numpy()
            attn_ts = np.zeros(T)
            counts = np.zeros(T)
            for i_w, start in enumerate(range(0, T - ATTN_WINDOW + 1, ATTN_STRIDE)):
                attn_ts[start:start+ATTN_WINDOW] += a[i_w]
                counts[start:start+ATTN_WINDOW] += 1
            counts[counts == 0] = 1
            attn_full[:, lead] = attn_ts / counts

        # reconstrucción completa de 12 leads x 5000 samples
        recon_m, recon_s = reconstruct_full_mean_std(model, sig_ex)
        n_leads = sig_ex.shape[0]
        t = np.arange(sig_ex.shape[1])


        x_orig_full = sig_ex.numpy().T
        x_mean_full = recon_m.T.cpu().numpy()
        x_std_full  = recon_s.T.cpu().numpy()
        mse_full = (x_orig_full - x_mean_full) ** 2
        var_full = np.log(x_std_full ** 2 + 1e-6)
        var_full = np.maximum(var_full, 0)
        anomaly_full = ALPHA * mse_full + (1 - ALPHA) * var_full

        idx_global = ids.index(rid_ex)
        pred_lbl = int(scores[idx_global] > thr)

        def plot_full_multilead_with_bpm(*args, **kwargs):
            orig_suptitle = plt.suptitle
            def new_suptitle(*a, **k):
                return orig_suptitle(
                    f'Epoch {idx_ex}: true={true_lbl}, pred={pred_lbl}, BPM={bpm_mean:.1f}',
                    fontsize=14, 
                    y=0.90
                )
            plt.suptitle = new_suptitle
            try:
                out_path = plot_full_multilead(*args, **kwargs)
            finally:
                plt.suptitle = orig_suptitle
            return out_path

        plot_full_multilead_with_bpm(
            x_orig_full,
            x_mean_full,
            x_std_full,
            epoch=idx_ex,
            out_dir_det=out_dir_full,
            out_dir_var=args.plot_dir,
            attn_full=attn_full,
            anomaly_full=anomaly_full,
            true_lbl=true_lbl,
            pred_lbl=pred_lbl
        )
        total_len = x_orig_full.shape[0]
        n_windows = (total_len - WINDOW) // STRIDE + 1
        window_sums = []
        for w in range(n_windows):
            start = w * STRIDE
            end   = start + WINDOW
            window_sums.append(anomaly_full[start:end, :].sum())
        best_w = int(np.argmax(window_sums))
        ws = best_w * STRIDE
        we = ws + WINDOW

        x_orig_win   = x_orig_full[ws:we, :]
        x_mean_win   = x_mean_full[ws:we, :]
        x_std_win    = x_std_full[ws:we, :]
        anomaly_win  = anomaly_full[ws:we, :]
        saliency_win = attn_full[ws:we, :] if 'attn_full' in locals() else None

        plot_full_multilead_with_bpm(
            x_orig_win, x_mean_win, x_std_win,
            epoch=idx_ex,
            out_dir_det=out_dir_windows,
            out_dir_var=args.plot_dir,
            attn_full=saliency_win,
            anomaly_full=anomaly_win,
            true_lbl=true_lbl,
            pred_lbl=pred_lbl
        )
        if unstable.size > 0:
            print(f"Sample {rid_ex}: unstable RR intervals at beats {unstable.tolist()}")

if __name__=='__main__':
    p = argparse.ArgumentParser()

    # MODEL PARAMETERS
    p.add_argument('--ckpt', default=str(PROJECT_ROOT / 'src' / 'weights' / 'best_vae_attn_model.pt'),
                help='Path to model checkpoint')

    # CPSC dataset paths
    p.add_argument('--cpsc_csv', type=str, default=str(PROJECT_ROOT / 'data' / 'inference_data' / 'cpsc' / 'processed_reference.csv'),
                help='Path to processed_reference.csv for CPSC evaluation')
    p.add_argument('--cpsc_dir', type=str, default=str(PROJECT_ROOT / 'data' / 'inference_data' / 'cpsc' / 'processed_cpsc'),
                help='Directory containing processed_cpsc .npy files')
    p.add_argument('--cpsc_max_samples', type=int, default=10,
                help='Maximum number of CPSC samples to use')

    # Number of example signals to plot
    p.add_argument('--examples', type=int, default=10,
                help='Number of example signals to plot')

    # Directory to save plots
    p.add_argument('--plot_dir', type=str, default=BASE_PLOT_DIR_VAE,
                help='Directory to save visualization plots')

    args = p.parse_args()
    main(args)