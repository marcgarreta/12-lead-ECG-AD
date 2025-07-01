
from pathlib import Path
import sys

THIS_FILE = Path(__file__).resolve()
SRC_DIR   = THIS_FILE.parent
ROOT_DIR  = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

WEIGHTS_DIR = SRC_DIR / "weights"
DATA_DIR    = ROOT_DIR / "data" / "inference_data"
CPSC_DIR    = DATA_DIR / "cpsc" / "processed_cpsc"
CPSC_CSV    = DATA_DIR / "cpsc" / "processed_reference.csv"
IMG_DIR     = ROOT_DIR / "img"
PLOTS_CAE_DIR = IMG_DIR / "plots_cae"
import os, math, json, argparse, numpy as np, pandas as pd, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from tqdm.auto import tqdm
import torch

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from models.cae import CAE

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
    # shape = [12, T]
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
    # Transpose to [T, 12] 
    sig_T = sig_arr.T  
    wins = []
    for start in range(0, T - WINDOW + 1, STRIDE):
        wins.append(sig_T[start:start+WINDOW])
    return torch.stack(wins) 

def reconstruct_full(model, sig):
    win_tensor = slide_windows(sig).to(DEVICE)          # [19,500,12]
    win_tensor = win_tensor.permute(0, 2, 1)
    with torch.no_grad():
        x_mean = model(win_tensor)
        x_mean = x_mean.permute(0, 2, 1)
    recon = torch.zeros((5000, 12), device=DEVICE)
    counts = torch.zeros((5000, 12), device=DEVICE)
    for i, start in enumerate(range(0, 5000 - WINDOW + 1, STRIDE)):
        recon[start:start+WINDOW] += x_mean[i]
        counts[start:start+WINDOW] += 1
    counts[counts == 0] = 1
    recon = recon / counts
    return recon.T.cpu()                                # [12,5000]

def reconstruct_full_mean_std(model, sig):
    full = reconstruct_full(model, sig)
    zeros = torch.zeros_like(full)
    return full, zeros


def ecg_score(model, sig):
    win_tensor = slide_windows(sig).to(DEVICE)
    win_in = win_tensor.permute(0, 2, 1)
    with torch.no_grad():
        recon_out = model(win_in)
    recon = recon_out.permute(0, 2, 1)
    err = (win_tensor - recon) ** 2
    return float(err.mean())

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
                        saliency_full: np.ndarray | None = None,
                        anomaly_full: np.ndarray | None = None,
                        true_lbl: int | None = None,
                        pred_lbl: int | None = None):

    os.makedirs(out_dir_det, exist_ok=True)
    T, n_leads = x_orig_full.shape
    t = np.arange(T)
    if saliency_full is not None:
        sal_min = float(np.nanmin(saliency_full))
        sal_max = float(np.nanmax(saliency_full))
        sal_p1 = np.percentile(saliency_full, 1)
        sal_p99 = np.percentile(saliency_full, 99)
    else:
        sal_min = sal_max = sal_p1 = sal_p99 = None

    mae_full = np.abs(x_orig_full - x_mean_full)
    mae_min = float(mae_full.min())
    mae_max = float(mae_full.max())
    if abs(mae_max - mae_min) < 1e-6:
        mae_min -= 1e-6
        mae_max += 1e-6

    if anomaly_full is not None:
        anomaly_min = float(np.nanmin(anomaly_full))
        anomaly_max = float(np.nanmax(anomaly_full))
        if not np.isfinite(anomaly_min) or not np.isfinite(anomaly_max):
            anomaly_min, anomaly_max = 0.0, 0.0
        elif abs(anomaly_max - anomaly_min) < 1e-6:
            anomaly_min -= 1e-6
            anomaly_max += 1e-6
    else:
        anomaly_min = anomaly_max = None

    # Remove global y-limits, use per-lead percentile-based scaling below
    fig = plt.figure(figsize=(14, 2.5 * n_leads), constrained_layout=True)
    gs = gridspec.GridSpec(n_leads * 4, 1,
                           height_ratios=[5,5,1,1] * n_leads,
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
        # Set per-lead y-limits based on 1st and 99th percentiles
        orig_p1 = np.nanpercentile(x_orig_full[:, i], 1)
        orig_p99 = np.nanpercentile(x_orig_full[:, i], 99)
        rec_p1 = np.nanpercentile(x_mean_full[:, i], 1)
        rec_p99 = np.nanpercentile(x_mean_full[:, i], 99)
        ymin = min(orig_p1, rec_p1)
        ymax = max(orig_p99, rec_p99)
        if not np.isfinite(ymin) or not np.isfinite(ymax) or abs(ymax - ymin) < 1e-6:
            ymin, ymax = -1, 1
        ax0.set_ylim(ymin, ymax)
        ax0.set_ylabel(f'Lead {i+1}')
        if i == 0:
            ax0.legend(['Original','Recon'], loc='upper right', fontsize='small')

        # 2) Anomaly score line (same height as ax0)
        ax1 = fig.add_subplot(gs[4*i+1], sharex=ax0)
        if anomaly_full is not None:
            ax1.plot(t, anomaly_full[:, i], color='green', lw=1)
            ax1.set_ylim(anomaly_min, anomaly_max)
            ax1.set_ylabel('AS')
            if i < n_leads-1:
                ax1.set_xticks([])
        else:
            ax1.axis('off')

        # 3) Saliency strip
        ax2 = fig.add_subplot(gs[4*i+2], sharex=ax0)
        if saliency_full is not None:
            # Smooth saliency with a Gaussian filter for smoother color transitions
            raw_sal = saliency_full[:, i]
            data_sal = gaussian_filter1d(raw_sal, sigma=2)
            # Plot with bilinear interpolation for smoothness
            ax2.imshow(
                data_sal[None, :], aspect='auto', cmap='viridis',
                vmin=sal_p1, vmax=sal_p99,
                extent=[t[0], t[-1], ax2.get_ylim()[0], ax2.get_ylim()[1]],
                interpolation='bilinear', zorder=0
            )
        else:
            ax2.axis('off')
        # ax2.set_ylabel('Saliency')  # Removed as per instructions
        if i < n_leads-1:
            ax2.set_xticks([])
        ax2.set_yticks([])

        # 4) MAE strip
        ax3 = fig.add_subplot(gs[4*i+3], sharex=ax0)
        mae = np.abs(x_orig_full[:, i] - x_mean_full[:, i])
        # Enhanced MAE strip with Reds colormap and 99th percentile cap
        mae_vmax = np.percentile(mae, 99)
        _plot_strip(ax3, mae, idx=0, cmap='Reds',
                    vmin=0, vmax=mae_vmax)
        # ax3.set_ylabel('MSE')  # Removed as per instructions
        if i < n_leads-1:
            ax3.set_xticks([])
        ax3.set_yticks([])

    # (Removed: Mark unstable heartbeats if provided)

    # Label X-axis on last subplot
    fig.get_axes()[-1].set_xlabel('Time')

    # (Legend for strips removed)

    plt.suptitle(f'Epoch {epoch}: true={true_lbl}, pred={pred_lbl}', fontsize=14)

    import matplotlib as mpl
    from matplotlib import cm
    # Use percentile-based sal_p1/sal_p99 for colorbar normalization
    if saliency_full is not None:
        sal_norm = mpl.colors.Normalize(vmin=sal_p1, vmax=sal_p99)
    else:
        sal_norm = None
    mae_norm = mpl.colors.Normalize(vmin=mae_min, vmax=mae_max)

    # Colorbar positions
    cbar_width = 0.015
    cbar_height = 0.18
    right_edge = 0.93
    # Saliency colorbar
    if saliency_full is not None:
        cax_sal = fig.add_axes([right_edge, 0.75, cbar_width, cbar_height])
        mpl.colorbar.ColorbarBase(cax_sal, cmap=mpl.colormaps['viridis'], norm=sal_norm, orientation='vertical')
        cax_sal.set_title('Saliency', fontsize=8)
    cax_mae = fig.add_axes([right_edge, 0.52, cbar_width, cbar_height])
    mpl.colorbar.ColorbarBase(cax_mae, cmap=mpl.colormaps['Reds'], norm=mae_norm, orientation='vertical')
    cax_mae.set_title('MAE', fontsize=8)
    out_path = os.path.join(out_dir_det, f'epoch_{epoch:03d}_full_multilead.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'➜ Guardado {out_path}')

@torch.no_grad()
def main(args):
    # Create output directories inside args.plot_dir
    out_dir_full = os.path.join(args.plot_dir, 'full_lead')
    out_dir_windows = os.path.join(args.plot_dir, 'windows')
    os.makedirs(out_dir_full, exist_ok=True)
    os.makedirs(out_dir_windows, exist_ok=True)
    import glob, pandas as pd, random
    samples = None
    if args.cpsc_dir and args.cpsc_csv:
        cpsc_df = pd.read_csv(args.cpsc_csv)
        cpsc_df['path'] = cpsc_df['Recording'].astype(str).apply(
            lambda r: os.path.join(args.cpsc_dir, f"{r}.npy"))
        cpsc_df = cpsc_df[cpsc_df.path.map(os.path.exists)]
        samples = [(row['path'], int(row['label']), row['Recording'])
                   for _, row in cpsc_df.iterrows()]
        if args.cpsc_max_samples is not None:
            samples = random.sample(samples, min(args.cpsc_max_samples, len(samples)))
    else:
        raise ValueError("Por favor especifique un directorio de visualización y su CSV")

    print(f"[DEBUG] Found {len(samples)} valid samples.")
    if len(samples) == 0:
        raise ValueError("No valid samples found. Check your dataset paths and CSV files.")

    # 1. modelo
    ckpt_data = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    model = CAE(in_channels=12).to(DEVICE)
    model.load_state_dict(ckpt_data if isinstance(ckpt_data, dict) else ckpt_data.state_dict())

    model.eval()
    examples = []  

    print()
    ids, ys, scores = [], [], []
    for idx_ex, (fp, true_lbl, rid_ex) in enumerate(samples, start=1):
        sig = torch.tensor(np.load(fp), dtype=torch.float32)
        if sig.ndim == 2 and sig.shape[0] != 12 and sig.shape[1] == 12:
            sig = sig.T
        elif sig.ndim == 2 and sig.shape[0] > 12 and sig.shape[1] > 12:
            print(f"Warning: unexpected signal shape {tuple(sig.shape)}, extracting first 12 rows")
            sig = sig[:12, :]
        elif sig.ndim != 2:
            raise ValueError(f"Expected 2D signal, got {sig.ndim}D")
        L = sig.shape[1]
        if L < 5000:
            # Padding
            sig = F.pad(sig, (0, 5000 - L))
        elif L > 5000:
            # Truncate
            sig = sig[:, :5000]

        if len(examples) < args.examples:
            examples.append({'sig': sig, 'rid': rid_ex, 'true': int(true_lbl)})

        ids.append(rid_ex)
        ys.append(int(true_lbl))
        scores.append(ecg_score(model, sig))

    ys = np.array(ys)
    scores = np.array(scores)

    print(f"[DEBUG] Number of scores computed: {len(scores)}")
    if len(scores) == 0:
        raise ValueError("No anomaly scores computed. Possible data processing issue.")

    # Compute threshold
    thr, _ = find_best_threshold(ys, scores)

    for idx_ex, ex in enumerate(examples, start=1):
        sig_ex  = ex['sig']
        rid_ex  = ex['rid']
        true_lbl = ex['true']

        # Uso de Lead II para detección de R-peaks
        ecg_ch = sig_ex[1].cpu().numpy()
        # Detección de R-peaks 
        peaks, _ = find_peaks(ecg_ch, distance=0.4 * 500)
        if len(peaks) >= 2:
            times = peaks / 500.0  
            rr = np.diff(times)
            bpm_inst = 60.0 / rr  
            bpm_mean = float(np.mean(bpm_inst))

            unstable = np.where(np.abs(bpm_inst - bpm_mean) > 10)[0]
        else:
            bpm_mean = float('nan')
            unstable = np.array([], dtype=int)

        unstable_times = (peaks[unstable+1] if len(unstable)>0 else np.array([],dtype=int))

        T = sig_ex.shape[1]
        n_leads = sig_ex.shape[0]
        saliency_full = np.zeros((T, n_leads))
        for lead in range(n_leads):

            sig_1ch = torch.zeros_like(sig_ex)
            sig_1ch[lead] = sig_ex[lead]
            # Saliency computation
            win_tensor = []
            sig_arr = sig_1ch

            for start in range(0, T - ATTN_WINDOW + 1, ATTN_STRIDE):
                win_tensor.append(sig_arr[:, start:start+ATTN_WINDOW].T)
            win_1ch = torch.stack(win_tensor).to(DEVICE)  
            win_1ch = win_1ch.permute(0, 2, 1)
            with torch.no_grad():
                attn_out = model(win_1ch)

            rec_win = attn_out.permute(0, 2, 1)  
            input_win = win_1ch.permute(0, 2, 1)   
            mse_local = ((input_win - rec_win) ** 2).mean(dim=(1,2)).cpu().numpy()
            a = mse_local
            sal_ts = np.zeros(T)
            counts = np.zeros(T)
            for i_w, start in enumerate(range(0, T - ATTN_WINDOW + 1, ATTN_STRIDE)):
                sal_ts[start:start+ATTN_WINDOW] += a[i_w]
                counts[start:start+ATTN_WINDOW] += 1
            counts[counts == 0] = 1
            saliency_full[:, lead] = sal_ts / counts

        recon_m, recon_s = reconstruct_full_mean_std(model, sig_ex)
        n_leads = sig_ex.shape[0]
        t = np.arange(sig_ex.shape[1])

        x_orig_full = sig_ex.numpy().T
        x_mean_full = recon_m.T.cpu().numpy()
        x_std_full  = recon_s.T.cpu().numpy()
        mse_full = (x_orig_full - x_mean_full) ** 2
        var_full = np.log(x_std_full ** 2 + 1e-6)
        anomaly_full = ALPHA * mse_full + (1 - ALPHA) * var_full

        idx_global = ids.index(rid_ex)
        pred_lbl = int(scores[idx_global] > thr)

        def plot_full_multilead_with_bpm(*args, **kwargs):
            orig_suptitle = plt.suptitle
            def new_suptitle(*a, **k):
                # Put BPM ()
                return orig_suptitle(
                    f'Epoch {idx_ex}: true={true_lbl}, pred={pred_lbl}, BPM={bpm_mean:.1f}',
                    fontsize=14
                )
            plt.suptitle = new_suptitle
            try:
                out = plot_full_multilead(*args, **kwargs)
            finally:
                plt.suptitle = orig_suptitle
            return out

        plot_full_multilead_with_bpm(
            x_orig_full,
            x_mean_full,
            x_std_full,
            epoch=idx_ex,
            out_dir_det=out_dir_full,
            out_dir_var=out_dir_full,
            saliency_full=saliency_full,
            anomaly_full=anomaly_full,
            true_lbl=true_lbl,
            pred_lbl=pred_lbl
        )

        win_tensor = slide_windows(sig_ex)  
        n_windows = win_tensor.shape[0]
        win_dev = win_tensor.to(DEVICE).permute(0, 2, 1)  
        with torch.no_grad():
            recon_win = model(win_dev).permute(0, 2, 1)  
        mae_per_window = torch.mean((win_tensor - recon_win) ** 2, dim=(1,2)).cpu().numpy()
        best_idx = int(np.argmax(mae_per_window))
        x_win_orig = win_tensor[best_idx].numpy()      
        x_win_recon = recon_win[best_idx].cpu().numpy()  
        sal_win = np.abs(x_win_orig - x_win_recon)      

        t_win = np.arange(WINDOW)
        fig_w = plt.figure(figsize=(14, 2.5 * 12), constrained_layout=True)
        gs_w = gridspec.GridSpec(12 * 3, 1, height_ratios=[5,1,1] * 12, hspace=0.2)
        for i in range(12):
            # Original vs reconstructed signal
            ax0 = fig_w.add_subplot(gs_w[3*i])
            ax0.plot(t_win, x_win_orig[:,i], color='black', lw=1)
            ax0.plot(t_win, x_win_recon[:,i], color='C1', lw=1, ls='--')
            ax0.set_ylabel(f'Lead {i+1}')
            if i==0:
                ax0.legend(['Orig','Recon'], loc='upper right', fontsize='small')
            
            # Saliecny 
            ax1 = fig_w.add_subplot(gs_w[3*i+1], sharex=ax0)
            ax1.imshow(sal_win[:,i][None,:], aspect='auto', cmap='viridis',
                       vmin=np.percentile(sal_win[:,i],1),
                       vmax=np.percentile(sal_win[:,i],99),
                       extent=[t_win[0], t_win[-1], 0, 1])
            ax1.axis('off')
            
            # MAE strip
            ax2 = fig_w.add_subplot(gs_w[3*i+2], sharex=ax0)
            ax2.imshow(sal_win[:,i][None,:], aspect='auto', cmap='Reds',
                       vmin=0, vmax=np.percentile(sal_win[:,i],99),
                       extent=[t_win[0], t_win[-1], 0, 1])
            ax2.axis('off')
        fig_w.suptitle(f'Window {best_idx} MAE-max (Epoch {idx_ex})', fontsize=14)
        out_win = os.path.join(out_dir_windows, f'epoch_{idx_ex:03d}_window_{best_idx}.png')
        fig_w.savefig(out_win, dpi=150)
        plt.close(fig_w)
        print(f'➜ Guardado ventana {best_idx} en {out_win}')
        # MSE
        win_tensor = slide_windows(sig_ex)  
        n_windows = win_tensor.shape[0]
        n_leads = win_tensor.shape[2]

        win_tensor_dev = win_tensor.to(DEVICE).permute(0, 2, 1) 
        with torch.no_grad():
            recon_win = model(win_tensor_dev)  
        recon_win = recon_win.permute(0, 2, 1)
        abs_err = torch.abs(win_tensor.to(recon_win.device) - recon_win)  
        mae_window_lead = abs_err.mean(dim=1).cpu().numpy()  
        # Plot of heatmap
        fig, ax = plt.subplots(figsize=(max(6, n_windows/3), 5))
        im = ax.imshow(mae_window_lead.T, aspect='auto', cmap='Reds', origin='lower')
        ax.set_xlabel('Window')
        ax.set_ylabel('Lead')
        ax.set_yticks(np.arange(n_leads))
        ax.set_yticklabels([f'Lead {i+1}' for i in range(n_leads)])
        cbar = fig.colorbar(im, ax=ax, label='MAE')
        ax.set_title(f'Per-window MAE (Epoch {idx_ex})')
        out_path = os.path.join(out_dir_windows, f'epoch_{idx_ex:03d}_windows.png')
        fig.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'➜ Guardado {out_path}')
        if unstable.size > 0:
            print(f"Sample {rid_ex}: unstable RR intervals at beats {unstable.tolist()}")

if __name__=='__main__':
    import argparse

    p = argparse.ArgumentParser()

    # MODEL PARAMETERS
    p.add_argument(
        '--ckpt',
        default=str(WEIGHTS_DIR / "best_cae_model.pth"),
        help='Path to model checkpoint'
    )

    # MAX SAMPLES
    p.add_argument('--cpsc_max_samples', type=int, default=10,
                   help='Máximo de muestras a usar de CPSC (antes de mezclar)')

    # Number of example signals to plot
    p.add_argument('--examples', type=int, default=10,
                   help='Número de señales a graficar como ejemplo')

    # CPSC dataset paths
    p.add_argument('--cpsc_csv', type=str, default=str(CPSC_CSV),
                   help='Path to processed_reference.csv for CPSC evaluation')
    p.add_argument('--cpsc_dir', type=str, default=str(CPSC_DIR),
                   help='Directory containing processed_cpsc .npy files')

    # Directory to save plots
    p.add_argument('--plot_dir', type=str,
                   default=str(PLOTS_CAE_DIR),
                   help='Directory to save visualization plots')

    args = p.parse_args()
    main(args)
