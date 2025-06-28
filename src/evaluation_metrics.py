import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from tqdm.auto import tqdm
import pandas as pd
import glob as glob
import os as os


from model_bilstm import VAE

WINDOW = 500
STRIDE = 250
ALPHA  = 0.7
BETA   = 0.3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def slide_windows(sig):
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

def ecg_score(model, sig):
    win = slide_windows(sig).to(DEVICE)
    x_mean, x_logvar, mu, logvar, attn, _ = model.forward(win)
    sigma2 = torch.exp(x_logvar)
    err = (win - x_mean)**2 / (sigma2 + 1e-6)
    kl = -0.5*(1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=[1,2])
    var_term = x_logvar
    base = ALPHA*err.sum([1,2]) + (1-ALPHA)*var_term.sum([1,2]) + BETA*kl
    return float(base.mean())

def find_best_threshold(y_true, y_scores):
    best, thr = -1, 0
    for t in np.percentile(y_scores, np.linspace(0,100,100)):
        yhat = (y_scores > t).astype(int)
        f1 = f1_score(y_true, yhat)
        if f1>best:
            best, thr = f1, t
    return thr, best

def main(args):

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    if isinstance(ckpt, dict):
        model = VAE(n_leads=12, n_latent=64).to(DEVICE)
        model.load_state_dict(ckpt)
    else:
        model = ckpt.to(DEVICE)
    model.eval()

    df = pd.read_csv(args.ptbxl_csv)
    id_col = next(c for c in ['Recording','ID','id','recording'] if c in df.columns)
    df['path'] = df[id_col].astype(str).apply(lambda r: os.path.join(args.ptbxl_data, f"{r}.npy"))
    df = df[df.path.map(os.path.exists)].reset_index(drop=True)

    # Subsample PTB-XL if requested
    ptbxl_df = df
    if args.ptbxl_max_samples:
        ptbxl_df = ptbxl_df.sample(n=min(args.ptbxl_max_samples, len(ptbxl_df)),
                                   random_state=42).reset_index(drop=True)

    # ——— Include CPSC dataset samples ———
    if args.cpsc_csv and args.cpsc_dir:
        cpsc_df = pd.read_csv(args.cpsc_csv)
        # Build paths
        cpsc_df['path'] = cpsc_df['Recording'].astype(str).apply(
            lambda r: os.path.join(args.cpsc_dir, f"{r}.npy"))
        # Keep only existing files
        cpsc_df = cpsc_df[cpsc_df['path'].apply(os.path.exists)].reset_index(drop=True)
        # Ensure label column is int
        cpsc_df['label'] = cpsc_df['label'].astype(int)
        # Rename Recording to match id_col
        cpsc_df = cpsc_df.rename(columns={'Recording': id_col})

        # Subsample CPSC if requested
        if args.cpsc_max_samples:
            cpsc_df = cpsc_df.sample(n=min(args.cpsc_max_samples, len(cpsc_df)),
                                     random_state=42).reset_index(drop=True)

    if args.mimic_dir:
        import glob as glob, os as os
        mimic_files = glob.glob(os.path.join(args.mimic_dir, '*.npy'))
        df_mimic = pd.DataFrame({
            id_col: [os.path.basename(f).replace('.npy','') for f in mimic_files],
            'path':  mimic_files,
            'label': 1
        })

        # Subsample MIMIC if requested
        if args.mimic_max_samples is not None:
            df_mimic = df_mimic.sample(n=min(args.mimic_max_samples, len(df_mimic)),
                                       random_state=42).reset_index(drop=True)

    frames = [ptbxl_df]
    if args.mimic_dir:
        frames.append(df_mimic)

    if args.cpsc_csv and args.cpsc_dir:
        frames.append(cpsc_df)
    df = pd.concat(frames, ignore_index=True)

    ys, scores = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Scoring'):
        sig = torch.tensor(np.load(row.path).T, dtype=torch.float32)
        # Ensure signal length is 5000 samples
        L = sig.shape[1]
        if L < 5000:
            sig = F.pad(sig, (0, 5000 - L))
        elif L > 5000:
            sig = sig[:, :5000]
        ys.append(int(row.get('label', row.label)))  # if original CSV had `label`
        scores.append(ecg_score(model, sig))

    ys     = np.array(ys)
    scores = np.array(scores)

    if len(np.unique(ys))<2:
        print("Solo una clase presente; no puedo calcular F1/Recall/AUC.")
    else:
        thr, best_f1 = find_best_threshold(ys,scores)
        recall = recall_score(ys, scores>thr)
        auc    = roc_auc_score(ys, scores)
        print(f"Threshold: {thr:.3f} (best F1={best_f1:.3f})")
        print(f"F1={best_f1:.3f}  Recall={recall:.3f}  AUC={auc:.3f}")

if __name__=='__main__':
    p = argparse.ArgumentParser()

    # MODEL PARAMETERS
    p.add_argument('--ckpt',        required=True,
                   help='Path to model checkpoint')

    # MAX SAMPLES 
    p.add_argument('--ptbxl_max_samples', type=int, required=True,
                   help='Máximo de muestras a usar de PTB-XL (antes de mezclar)')
    p.add_argument('--cpsc_max_samples',  type=int, required=True,
                   help='Máximo de muestras a usar de CPSC (antes de mezclar)')
    p.add_argument('--mimic_max_samples', type=int, required=True,
                   help='Máximo de muestras a usar de MIMIC (antes de mezclar)')

    # CPSC dataset paths
    p.add_argument('--cpsc_csv', type=str, required=True,
                   help='Path to processed_reference.csv for CPSC evaluation')
    p.add_argument('--cpsc_dir', type=str, required=True,
                   help='Directory containing processed_cpsc .npy files')
    # PTB-XL dataset paths
    p.add_argument('--ptbxl_data',  required=True)
    p.add_argument('--ptbxl_csv',   required=True,
                   help='CSV file for PTB-XL dataset')
    # MIMIC dataset paths
    p.add_argument('--mimic_dir', type=str, required=True,
                   help='Directory containing MIMIC abnormal .npy files (all label=1)')

    args = p.parse_args()
    main(args)

