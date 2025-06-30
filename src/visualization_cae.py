
from pathlib import Path
import sys

# Resolve project root and add src to path
THIS_FILE = Path(__file__).resolve()
SRC_DIR   = THIS_FILE.parent
ROOT_DIR  = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

# Standard directory locations
WEIGHTS_DIR = SRC_DIR / "weights"
DATA_DIR    = ROOT_DIR / "data" / "inference_data"
CPSC_DIR    = DATA_DIR / "cpsc" / "processed_cpsc"
CPSC_CSV    = DATA_DIR / "cpsc" / "processed_reference.csv"
MIMIC_DIR   = DATA_DIR / "mimic_npy_abnormal"
PTBXL_DIR   = DATA_DIR / "ptbxl_zscore_200"
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
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from models.cae import CAE_M

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
    # Ensure tensor
    if not isinstance(sig, torch.Tensor):
        sig = torch.tensor(sig, dtype=torch.float32)
    # Ensure shape [12, T]
    if sig.ndim != 2:
        raise ValueError(f"Expected 2D signal, got {sig.ndim}D")
    if sig.shape[0] == 12:
        sig_arr = sig
    elif sig.shape[1] == 12:
        sig_arr = sig.T
    else:
        # fallback: if both dims >12, take first 12 rows
        if sig.shape[0] > 12:
            sig_arr = sig[:12, :]
        elif sig.shape[1] > 12:
            sig_arr = sig[:, :12].T
        else:
            raise ValueError(f"Unexpected signal shape {sig.shape}, expected one dimension = 12")
    # Now sig_arr is [12, T]
    T = sig_arr.shape[1]
    # Check length
    if T < WINDOW:
        sig_arr = torch.nn.functional.pad(sig_arr, (0, WINDOW - T))
        T = WINDOW
    # Transpose to [T, 12] for windowing
    sig_T = sig_arr.T  # [T,12]
    wins = []
    for start in range(0, T - WINDOW + 1, STRIDE):
        wins.append(sig_T[start:start+WINDOW])
    return torch.stack(wins)  # [n_windows, WINDOW, 12]

def reconstruct_full(model, sig):
    win_tensor = slide_windows(sig).to(DEVICE)          # [19,500,12]
    # reshape to [batch, channels, length] for CAE
    win_tensor = win_tensor.permute(0, 2, 1)
    with torch.no_grad():
        # model returns [batch, channels, length]
        x_mean = model(win_tensor)
        # reshape back to [batch, length, channels]
        x_mean = x_mean.permute(0, 2, 1)
    # superponer con stride 250
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
    # reshape for CAE
    win_in = win_tensor.permute(0, 2, 1)
    with torch.no_grad():
        recon_out = model(win_in)
    # reshape recon back
    recon = recon_out.permute(0, 2, 1)
    err = (win_tensor - recon) ** 2
    return float(err.mean())

def find_best_threshold(y_true, y_scores):
    best_f1, best_thr = -1, 0
    # reduce grid to 50 candidate thresholds for speed
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
    # Compute vmin/vmax for saliency and MSE strips
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

    # Compute vmin/vmax for anomaly score
    if anomaly_full is not None:
        # Compute anomaly limits ignoring NaN/Inf
        anomaly_min = float(np.nanmin(anomaly_full))
        anomaly_max = float(np.nanmax(anomaly_full))
        if not np.isfinite(anomaly_min) or not np.isfinite(anomaly_max):
            anomaly_min, anomaly_max = 0.0, 0.0
        elif abs(anomaly_max - anomaly_min) < 1e-6:
            anomaly_min -= 1e-6
            anomaly_max += 1e-6
    else:
        anomaly_min = anomaly_max = None

    # --------- Compute common y-axis limits for signal plots ---------
    # Include original signal and reconstruction ±2σ
    # Compute signal limits, ignoring NaN/Inf
    sig_min = float(np.nanmin(x_orig_full))
    sig_max = float(np.nanmax(x_orig_full))
    if not np.isfinite(sig_min) or not np.isfinite(sig_max):
        sig_min, sig_max = 0.0, 0.0
    # For variability plot, include ±2σ bounds
    rec_min = float((x_mean_full - 2*x_std_full).min())
    rec_max = float((x_mean_full + 2*x_std_full).max())
    sig_min = min(sig_min, rec_min)
    sig_max = max(sig_max, rec_max)
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
        # Set common y-limits for all leads
        ax0.set_ylim(sig_min, sig_max)
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
    # MAE colorbar
    cax_mae = fig.add_axes([right_edge, 0.52, cbar_width, cbar_height])
    mpl.colorbar.ColorbarBase(cax_mae, cmap=mpl.colormaps['Reds'], norm=mae_norm, orientation='vertical')
    cax_mae.set_title('MAE', fontsize=8)
    # plt.tight_layout(rect=[0, 0, 1, 0.96])  # Removed due to constrained_layout
    out_path = os.path.join(out_dir_det, f'epoch_{epoch:03d}_full_multilead.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'➜ Guardado {out_path}')

@torch.no_grad()
def main(args):
    PLOTS_CAE_DIR.mkdir(parents=True, exist_ok=True)
    # Define subdirectories for full and window-based plots
    FULL_DIR = PLOTS_CAE_DIR / "full"
    WINDOW_DIR = PLOTS_CAE_DIR / "windows"
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    WINDOW_DIR.mkdir(parents=True, exist_ok=True)
    import glob, os, pandas as pd, random
    # Build list of (file_path, label, record_id)
    samples = None
    # Priority: MIMIC, then CPSC, then PTB-XL
    if args.mimic_dir:
        files = glob.glob(os.path.join(args.mimic_dir, '*.npy'))
        samples = [(f, 1, os.path.basename(f).replace('.npy','')) for f in files]
        # Subsample if mimic_max_samples is set
        if args.mimic_max_samples is not None:
            samples = random.sample(samples, min(args.mimic_max_samples, len(samples)))
    elif args.cpsc_dir and args.cpsc_csv:
        cpsc_df = pd.read_csv(args.cpsc_csv)
        cpsc_df['path'] = cpsc_df['Recording'].astype(str).apply(
            lambda r: os.path.join(args.cpsc_dir, f"{r}.npy"))
        cpsc_df = cpsc_df[cpsc_df.path.map(os.path.exists)]
        samples = [(row['path'], int(row['label']), row['Recording'])
                   for _, row in cpsc_df.iterrows()]
        if args.cpsc_max_samples is not None:
            samples = random.sample(samples, min(args.cpsc_max_samples, len(samples)))
    elif args.ptbxl_data and args.ptbxl_csv:
        ptbxl_df = pd.read_csv(args.ptbxl_csv)
        ptbxl_df['path'] = ptbxl_df['Recording'].astype(str).apply(
            lambda r: os.path.join(args.ptbxl_data, f"{r}.npy"))
        ptbxl_df = ptbxl_df[ptbxl_df.path.map(os.path.exists)]
        samples = [(row['path'],
                    int(row['label']) if 'label' in ptbxl_df.columns else 0,
                    row['Recording'])
                   for _, row in ptbxl_df.iterrows()]
        if args.ptbxl_max_samples is not None:
            samples = random.sample(samples, min(args.ptbxl_max_samples, len(samples)))
    else:
        raise ValueError("Por favor especifique un directorio de visualización y su CSV")

    # 1. modelo
    ckpt_data = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    model = CAE_M(in_channels=12).to(DEVICE)
    model.load_state_dict(ckpt_data if isinstance(ckpt_data, dict) else ckpt_data.state_dict())

    model.eval()
    examples = []  # almacenará hasta args.examples ECGs para graficar

    print()
    ids, ys, scores = [], [], []
    for idx_ex, (fp, true_lbl, rid_ex) in enumerate(samples, start=1):
        sig = torch.tensor(np.load(fp), dtype=torch.float32)
        # Ensure sig has shape [12, T]
        if sig.ndim == 2 and sig.shape[0] != 12 and sig.shape[1] == 12:
            sig = sig.T
        elif sig.ndim == 2 and sig.shape[0] > 12 and sig.shape[1] > 12:
            # fallback: take first 12 channels
            print(f"Warning: unexpected signal shape {tuple(sig.shape)}, extracting first 12 rows")
            sig = sig[:12, :]
        elif sig.ndim != 2:
            raise ValueError(f"Expected 2D signal, got {sig.ndim}D")
        L = sig.shape[1]
        if L < 5000:
            # Padding
            sig = F.pad(sig, (0, 5000 - L))
        elif L > 5000:
            # Recortamos al principio para quedarnos con las 5000 primeras
            sig = sig[:, :5000]

        # guardar algunas señales para ejemplos de visualización
        if len(examples) < args.examples:
            examples.append({'sig': sig, 'rid': rid_ex, 'true': int(true_lbl)})

        ids.append(rid_ex)
        ys.append(int(true_lbl))
        scores.append(ecg_score(model, sig))

    ys = np.array(ys)
    scores = np.array(scores)

    # Compute threshold for predictions
    thr, _ = find_best_threshold(ys, scores)

    for idx_ex, ex in enumerate(examples, start=1):
        sig_ex  = ex['sig']
        rid_ex  = ex['rid']
        true_lbl = ex['true']

        # --- Compute BPM and detect unstable intervals ---
        # Use Lead II (index 1) for R-peak detection
        ecg_ch = sig_ex[1].cpu().numpy()
        # Detect R-peaks (min distance 0.4s at 500Hz)
        peaks, _ = find_peaks(ecg_ch, distance=0.4 * 500)
        if len(peaks) >= 2:
            times = peaks / 500.0  # seconds
            rr = np.diff(times)
            bpm_inst = 60.0 / rr   # instantaneous BPM
            bpm_mean = float(np.mean(bpm_inst))
            # find unstable where deviation >10 BPM
            unstable = np.where(np.abs(bpm_inst - bpm_mean) > 10)[0]
        else:
            bpm_mean = float('nan')
            unstable = np.array([], dtype=int)

        # convert unstable rr indices to sample times (use peaks array)
        unstable_times = (peaks[unstable+1] if len(unstable)>0 else np.array([],dtype=int))

        # Compute per-lead saliency by isolating each channel
        T = sig_ex.shape[1]
        n_leads = sig_ex.shape[0]
        saliency_full = np.zeros((T, n_leads))
        for lead in range(n_leads):
            # zero out other leads
            sig_1ch = torch.zeros_like(sig_ex)
            sig_1ch[lead] = sig_ex[lead]
            # Compute saliency with finer windows
            win_tensor = []
            sig_arr = sig_1ch
            # slide with ATTN_WINDOW and ATTN_STRIDE
            for start in range(0, T - ATTN_WINDOW + 1, ATTN_STRIDE):
                win_tensor.append(sig_arr[:, start:start+ATTN_WINDOW].T)
            win_1ch = torch.stack(win_tensor).to(DEVICE)  # [n_blocks, ATTN_WINDOW, 12]
            # reshape for CAE
            win_1ch = win_1ch.permute(0, 2, 1)
            with torch.no_grad():
                attn_out = model(win_1ch)
                # no recon needed here, extract saliency if model returns it;
                # if CAE only returns recon, skip saliency extraction
            # If model returns only recon, attn_out is [n_blocks, 12, ATTN_WINDOW]
            # For compatibility, mimic saliency as mean of recon error per window
            # (This is a placeholder; adjust as needed for true saliency)
            # Compute per-window mean squared error as pseudo-saliency
            rec_win = attn_out.permute(0, 2, 1)  # [n_blocks, ATTN_WINDOW, 12]
            input_win = win_1ch.permute(0, 2, 1)  # [n_blocks, ATTN_WINDOW, 12]
            mse_local = ((input_win - rec_win) ** 2).mean(dim=(1,2)).cpu().numpy()
            a = mse_local
            # reconstruct timeline for this lead
            sal_ts = np.zeros(T)
            counts = np.zeros(T)
            for i_w, start in enumerate(range(0, T - ATTN_WINDOW + 1, ATTN_STRIDE)):
                sal_ts[start:start+ATTN_WINDOW] += a[i_w]
                counts[start:start+ATTN_WINDOW] += 1
            counts[counts == 0] = 1
            saliency_full[:, lead] = sal_ts / counts

        # reconstrucción completa
        recon_m, recon_s = reconstruct_full_mean_std(model, sig_ex)
        n_leads = sig_ex.shape[0]
        t = np.arange(sig_ex.shape[1])

        # Prepare inputs for full-multilead plot
        # transpose to (T,12)
        x_orig_full = sig_ex.numpy().T
        x_mean_full = recon_m.T.cpu().numpy()
        x_std_full  = recon_s.T.cpu().numpy()
        # per-timestep anomaly score (MSE + var term)
        mse_full = (x_orig_full - x_mean_full) ** 2
        var_full = np.log(x_std_full ** 2 + 1e-6)
        anomaly_full = ALPHA * mse_full + (1 - ALPHA) * var_full

        idx_global = ids.index(rid_ex)
        pred_lbl = int(scores[idx_global] > thr)

        # Plot with BPM in title
        def plot_full_multilead_with_bpm(*args, **kwargs):
            # Patch the plt.suptitle call inside plot_full_multilead
            # We monkeypatch plt.suptitle temporarily
            orig_suptitle = plt.suptitle
            def new_suptitle(*a, **k):
                # Replace with BPM in title
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
            out_dir_det=str(FULL_DIR),
            out_dir_var=str(FULL_DIR),
            saliency_full=saliency_full,
            anomaly_full=anomaly_full,
            true_lbl=true_lbl,
            pred_lbl=pred_lbl
        )
        # --- Window-based full-multilead visualization for the window with highest anomaly score ---
        # Identify window index with highest per-window anomaly (MAE) score
        win_tensor = slide_windows(sig_ex)  # [n_windows, WINDOW, 12]
        n_windows = win_tensor.shape[0]
        # compute MAE per window
        win_dev = win_tensor.to(DEVICE).permute(0, 2, 1)  # [n_windows, 12, WINDOW]
        with torch.no_grad():
            recon_win = model(win_dev).permute(0, 2, 1)  # [n_windows, WINDOW, 12]
        mae_per_window = torch.mean((win_tensor - recon_win) ** 2, dim=(1,2)).cpu().numpy()
        best_idx = int(np.argmax(mae_per_window))
        # extract signals for best window
        x_win_orig = win_tensor[best_idx].numpy()      # [WINDOW,12]
        x_win_recon = recon_win[best_idx].cpu().numpy()  # [WINDOW,12]
        # compute saliency per lead in this window using MAE
        sal_win = np.abs(x_win_orig - x_win_recon)     # [WINDOW,12]
        # prepare time axis
        t_win = np.arange(WINDOW)
        # plot multi-lead for this window
        fig_w = plt.figure(figsize=(14, 2.5 * 12), constrained_layout=True)
        gs_w = gridspec.GridSpec(12 * 3, 1, height_ratios=[5,1,1] * 12, hspace=0.2)
        for i in range(12):
            # original & recon
            ax0 = fig_w.add_subplot(gs_w[3*i])
            ax0.plot(t_win, x_win_orig[:,i], color='black', lw=1)
            ax0.plot(t_win, x_win_recon[:,i], color='C1', lw=1, ls='--')
            ax0.set_ylabel(f'Lead {i+1}')
            if i==0:
                ax0.legend(['Orig','Recon'], loc='upper right', fontsize='small')
            # saliency strip (MAE)
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
        out_win = WINDOW_DIR / f'epoch_{idx_ex:03d}_window_{best_idx}.png'
        fig_w.savefig(out_win, dpi=150)
        plt.close(fig_w)
        print(f'➜ Guardado ventana {best_idx} en {out_win}')
        # --- Window-based MAE heatmap visualization ---
        # Compute windows
        win_tensor = slide_windows(sig_ex)  # [n_windows, WINDOW, 12]
        n_windows = win_tensor.shape[0]
        n_leads = win_tensor.shape[2]
        # Move to device and permute for model
        win_tensor_dev = win_tensor.to(DEVICE).permute(0, 2, 1)  # [n_windows, 12, WINDOW]
        with torch.no_grad():
            recon_win = model(win_tensor_dev)  # [n_windows, 12, WINDOW]
        # Permute back to [n_windows, WINDOW, 12]
        recon_win = recon_win.permute(0, 2, 1)
        # Compute MAE for each window and lead, average over time axis
        abs_err = torch.abs(win_tensor.to(recon_win.device) - recon_win)  # [n_windows, WINDOW, 12]
        mae_window_lead = abs_err.mean(dim=1).cpu().numpy()  # [n_windows, 12]
        # Plot heatmap: x-axis windows, y-axis leads
        fig, ax = plt.subplots(figsize=(max(6, n_windows/3), 5))
        im = ax.imshow(mae_window_lead.T, aspect='auto', cmap='Reds', origin='lower')
        ax.set_xlabel('Window')
        ax.set_ylabel('Lead')
        ax.set_yticks(np.arange(n_leads))
        ax.set_yticklabels([f'Lead {i+1}' for i in range(n_leads)])
        cbar = fig.colorbar(im, ax=ax, label='MAE')
        ax.set_title(f'Per-window MAE (Epoch {idx_ex})')
        out_path = WINDOW_DIR / f'epoch_{idx_ex:03d}_windows.png'
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
    p.add_argument('--ptbxl_max_samples', type=int, default=10,
                   help='Máximo de muestras a usar de PTB-XL (antes de mezclar)')
    p.add_argument('--cpsc_max_samples', type=int, default=10,
                   help='Máximo de muestras a usar de CPSC (antes de mezclar)')
    p.add_argument('--mimic_max_samples', type=int, default=10,
                   help='Máximo de muestras a usar de MIMIC (antes de mezclar)')

    # Number of example signals to plot
    p.add_argument('--examples', type=int, default=10,
                   help='Número de señales a graficar como ejemplo')

    # CPSC dataset paths
    p.add_argument('--cpsc_csv', type=str, default=str(CPSC_CSV),
                   help='Path to processed_reference.csv for CPSC evaluation')
    p.add_argument('--cpsc_dir', type=str, default=str(CPSC_DIR),
                   help='Directory containing processed_cpsc .npy files')

    # PTB-XL dataset paths
    p.add_argument('--ptbxl_data', type=str, default=str(PTBXL_DIR),
                   help='Directory containing PTB-XL .npy files')
    p.add_argument('--ptbxl_csv', type=str, default=str(PTBXL_DIR / "labels.csv"),
                   help='CSV file for PTB-XL dataset')

    # MIMIC dataset paths
    p.add_argument('--mimic_dir', type=str, default=str(MIMIC_DIR),
                   help='Directory containing MIMIC abnormal .npy files (all label=1)')

    # Directory to save plots
    p.add_argument('--plot_dir', type=str,
                   default=str(PLOTS_CAE_DIR),
                   help='Directory to save visualization plots')

    args = p.parse_args()
    main(args)
