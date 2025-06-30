import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score, roc_auc_score, confusion_matrix, average_precision_score
from tqdm.auto import tqdm
import pandas as pd
import glob as glob
import os as os

import torch.serialization

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / 'src'))
CPSC_CSV_DEFAULT = ROOT / 'data' / 'inference_data' / 'cpsc' / 'processed_reference_balanced.csv'
CPSC_DIR_DEFAULT = ROOT / 'data' / 'inference_data' / 'cpsc' / 'processed_cpsc'
WEIGHTS_DIR = ROOT / 'src' / 'weights'
VAE_CKPT_DEFAULT = WEIGHTS_DIR / 'best_vae_attn_model.pt'
CAE_CKPT_DEFAULT = WEIGHTS_DIR / 'best_cae_model.pth'

from models.vae_bilstm_attention import VAE
from models.cae import CAE
from models.vae_bilstm_mha import VAE_BILSTM_MHA, VAEEncoder, VAEDecoder, GaussianNoise, MHA

WINDOW = 500
STRIDE = 250
ALPHA  = 0.4
BETA   = 0.6
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# From signal to windowed segments
def slide_windows(sig):
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
    sig_T = sig_arr.T  
    wins = []
    for start in range(0, T - WINDOW + 1, STRIDE):
        wins.append(sig_T[start:start+WINDOW])
    return torch.stack(wins)  # [n_windows, WINDOW, 12]

def ecg_score(model, sig):
    if isinstance(model, CAE):
        # CAE = MSE reconstruction error
        sig_t = sig.clone().detach()
        L = sig_t.shape[1]
        if L < 5000:
            sig_t = F.pad(sig_t, (0, 5000 - L))
        else:
            sig_t = sig_t[:, :5000]
        with torch.no_grad():
            recon = model(sig_t.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
        return float(((sig_t - recon)**2).mean())

    win = slide_windows(sig).to(DEVICE)
    out = model.forward(win)

    if len(out) == 6:
        x_mean, x_logvar, mu, logvar, attn, _ = out
    else: # VAE-based model

        x_mean, x_logvar, _, mu, logvar, attn, _ = out
    sigma2 = torch.exp(x_logvar)
    err = (win - x_mean)**2 / (sigma2 + 1e-6)
    kl = -0.5*(1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=[1,2])
    rec_err = err.sum([1,2])
    var_term = x_logvar.sum([1,2])
    base = ALPHA * rec_err + (1-ALPHA) * var_term + BETA * kl
    return float(base.mean())

def find_best_threshold(y_true, y_scores):
    candidates = np.linspace(y_scores.min(), y_scores.max(), 1000)
    best_f1 = -1.0
    best_thr = candidates[0]
    for t in candidates:
        yhat = (y_scores > t).astype(int)
        f1 = f1_score(y_true, yhat)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = t
    return best_thr, best_f1

def main(args):
    if args.model_type == 'ma_vae':
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            alt_path = Path(str(ckpt_path).replace('/export/fhome', '/fhome'))
            if alt_path.exists():
                ckpt_path = alt_path
            else:
                raise FileNotFoundError(f"Checkpoint not found at {args.ckpt} or {alt_path}")
        args.ckpt = str(ckpt_path)
        torch.serialization.add_safe_globals([VAE_BILSTM_MHA, VAEEncoder, VAEDecoder, GaussianNoise, MHA])
        ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)

        model = VAE_BILSTM_MHA(seq_len=WINDOW, n_leads=12, latent_dim=64).to(DEVICE)
        if isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        else:
            model = ckpt.to(DEVICE)
        model.eval()

    elif args.model_type == 'vae':
        vae_ckpt = torch.load(args.ckpt, map_location=DEVICE)
        model = VAE(n_leads=12, n_latent=64).to(DEVICE)
        if isinstance(vae_ckpt, dict):
            model.load_state_dict(vae_ckpt)
        else:
            model = vae_ckpt.to(DEVICE)
        model.eval()

    else: # CAE model
        cae_ckpt = torch.load(args.ckpt, map_location=DEVICE)
        model = CAE().to(DEVICE)
        if isinstance(cae_ckpt, dict):
            model.load_state_dict(cae_ckpt)
        else:
            try:
                model.load_state_dict(cae_ckpt.state_dict())
            except:
                model.load_state_dict(cae_ckpt)
        model.eval()

    # Use CPSC dataset for evaluation
    if args.cpsc_csv and args.cpsc_dir:
        cpsc_df = pd.read_csv(args.cpsc_csv)
        cpsc_df['path'] = cpsc_df['Recording'].astype(str).apply(
            lambda r: os.path.join(args.cpsc_dir, f"{r}.npy"))
        cpsc_df = cpsc_df[cpsc_df['path'].apply(os.path.exists)].reset_index(drop=True)
        cpsc_df['label'] = cpsc_df['label'].astype(int)
        id_col = 'Recording'
        cpsc_df = cpsc_df.rename(columns={'Recording': id_col})
        # Subsample cpsc dataset
        if args.cpsc_max_samples is not None:
            cpsc_df = cpsc_df.sample(n=min(args.cpsc_max_samples, len(cpsc_df)),
                                     random_state=42).reset_index(drop=True)
    df = cpsc_df

    ys, scores_vae = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Scoring'):
        sig = torch.tensor(np.load(row.path).T, dtype=torch.float32)
        L = sig.shape[1]
        if L < 5000:
            sig = F.pad(sig, (0, 5000 - L))
        elif L > 5000:
            sig = sig[:, :5000]
        ys.append(int(row.get('label', row.label))) 
        scores_vae.append(ecg_score(model, sig))

    ys = np.array(ys)
    scores_vae = np.array(scores_vae)

    # Min-max normalize scores
    sv = scores_vae
    scores_norm = (sv - sv.min()) / (sv.max() - sv.min() + 1e-8)
    scores_vae = scores_norm

    if len(np.unique(ys)) < 2:
        print("Only one class")
    else:
        thr, best_f1 = find_best_threshold(ys, scores_vae)
        pred = (scores_vae > thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(ys, pred).ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        auc = roc_auc_score(ys, scores_vae)
        pr_auc = average_precision_score(ys, scores_vae)
        print(f"Threshold: {thr:.6f} (best F1={best_f1:.3f})")
        print(f"Precision={precision:.3f}  Recall={recall:.3f}  AUC={auc:.3f}  PR AUC={pr_auc:.3f}")

if __name__=='__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--ckpt', type=str, default=None, required=False,
                   help='Path to model checkpoint (defaults to src/weights based on model_type)')
    p.add_argument('--model_type', choices=['vae','cae','ma_vae'], default='vae',
                   help='Model type: vae for VAE-BiLSTM-MHA, cae for convolutional autoencoder, ma_vae for the MA-VAE model')

    p.add_argument('--cpsc_csv', type=str,
                   default=str(CPSC_CSV_DEFAULT),
                   help='Path to processed_reference.csv for CPSC evaluation')
    p.add_argument('--cpsc_dir', type=str,
                   default=str(CPSC_DIR_DEFAULT),
                   help='Directory containing processed_cpsc .npy files')
    p.add_argument('--cpsc_max_samples', type=int, default=None,
                   help='Maximum number of CPSC samples to evaluate (for quick tests)')

    args = p.parse_args()

    if args.ckpt is None:
        weights_dir = ROOT / 'src' / 'weights'
        if args.model_type == 'vae':
            args.ckpt = str(weights_dir / 'best_vae_attn_model.pt')
        elif args.model_type == 'cae':
            args.ckpt = str(weights_dir / 'best_cae_model.pth')
        else:  # use of vae-bilstm-mha
            args.ckpt = str(ROOT / 'src' / 'MA_VAE' / 'checkpoints' / 'best_ma_vae_full.pt')

    main(args)

