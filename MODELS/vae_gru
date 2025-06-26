import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Linear, PReLU

class OmniAnomalyECG(nn.Module):
    def __init__(self, n_leads=12, window_size=250,
                 n_hidden=64, n_latent=16, n_layers=2):
        super().__init__()
        self.n_latent = n_latent
        # convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv1d(n_leads, n_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(n_hidden, n_hidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.gru = nn.GRU(n_hidden, n_hidden, n_layers,
                          batch_first=True, bidirectional=False)
        # encoder por timestep:
        self.enc = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.PReLU(),
            nn.Linear(n_hidden, 2 * n_latent)  # [μ, logσ²]
        )
        # decoder
        self.dec = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.PReLU(),
            nn.Linear(n_hidden, 2 * n_leads)
        )

    def forward(self, x):
        
        # x: (B, W, n_leads)
        B, W, _ = x.shape
        # apply convolutional layers
        x_conv = x.permute(0, 2, 1)       # (B, n_leads, W)
        x_conv = self.conv(x_conv)       # (B, n_hidden, W)
        x = x_conv.permute(0, 2, 1)      # (B, W, n_hidden)
        h_seq, _ = self.gru(x)           # (B, W, n_hidden)

        # codificar cada timestep
        stats = self.enc(h_seq)         # (B, W, 2*n_latent)
        μ, logvar = stats.chunk(2, dim=-1)
        σ = torch.exp(0.5 * logvar)
        ε = torch.randn_like(σ)
        z = μ + ε * σ                    # (B, W, n_latent)

        # decoder outputs per-lead mean and log-variance
        dec_stats = self.dec(z)         # (B, W, 2*n_leads)
        x_mean, x_logvar = dec_stats.chunk(2, dim=-1)
        return x_mean, x_logvar, z, μ, logvar
