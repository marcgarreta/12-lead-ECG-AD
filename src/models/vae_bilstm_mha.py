import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal

# Inspired by OmniAnomaly (KDD ’19) and MA-VAE (Multi-head Attention-based VAE; arXiv:2309.02253)

# Loss Function
def loss_function(x, x_mean, x_logvar, z_mean, z_logvar, beta):
    dist = Normal(x_mean, torch.exp(0.5 * x_logvar))
    log_px = dist.log_prob(x).sum(dim=[1, 2])
    recon = -log_px.mean()
    kl = 0.5 * (torch.exp(z_logvar) + z_mean**2 - 1 - z_logvar).sum(dim=[1, 2]).mean()
    return recon + beta * kl, recon, kl

# Beta Scheduler
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
    
# Gaussian Noise 
class GaussianNoise(nn.Module):
    def __init__(self, std=0.01):
        super().__init__()
        self.std = std
    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x

# Encoder
class VAEEncoder(nn.Module):
    def __init__(self, seq_len, n_leads, latent_dim):
        super().__init__()
        self.noise = GaussianNoise(0.01)
        self.lstm1 = nn.LSTM(n_leads, 512, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(512*2, 256, batch_first=True, bidirectional=True)
        self.linear_mean = nn.Linear(256*2, latent_dim)
        self.linear_logvar = nn.Linear(256*2, latent_dim)

    def forward(self, x):
        x = self.noise(x)
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        z_mean = self.linear_mean(out)
        z_logvar = self.linear_logvar(out)
        eps = torch.randn_like(z_mean)
        z = z_mean + torch.exp(0.5 * z_logvar) * eps
        return z_mean, z_logvar, z, out

# Multi-head Attention
class MHA(nn.Module):
    def __init__(self, n_leads, latent_dim, n_heads=8):
        super().__init__()
        self.to_q = nn.Linear(n_leads, latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, num_heads=n_heads, batch_first=True)

    def forward(self, x, z):
        q = self.to_q(x)
        attn_out, attn_weights = self.attn(q, q, z, need_weights=True)
    
        return attn_out, attn_weights   

# Decoder
class VAEDecoder(nn.Module):
    def __init__(self, seq_len, n_leads, latent_dim):
        super().__init__()
        self.lstm1 = nn.LSTM(latent_dim, 256, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256*2, 512, batch_first=True, bidirectional=True)
        self.linear_mean = nn.Linear(512*2, n_leads)
        self.linear_logvar = nn.Linear(512*2, n_leads)

    def forward(self, a):
        out, _ = self.lstm1(a)
        out, _ = self.lstm2(out)
        x_mean = self.linear_mean(out)
        x_logvar = self.linear_logvar(out)
        eps = torch.randn_like(x_mean)
        x_recon = x_mean + torch.exp(0.5 * x_logvar) * eps
        return x_mean, x_logvar, x_recon

# VAE-BiLSTM-MHA Model
class VAE_BILSTM_MHA(nn.Module):
    def __init__(self, seq_len, n_leads, latent_dim):
        super().__init__()
        self.encoder = VAEEncoder(seq_len, n_leads, latent_dim)
        self.mha = MHA(n_leads, latent_dim)
        self.decoder = VAEDecoder(seq_len, n_leads, latent_dim)

    def forward(self, x):
        # Encode
        z_mean, z_logvar, z, _ = self.encoder(x)
        # Multi-head attention returns output + weights
        attn_out, attn_weights = self.mha(x, z)
        # Decode
        x_mean, x_logvar, x_recon = self.decoder(attn_out)
        # Focus score: mean absolute reconstruction error per sample
        focus = torch.mean(torch.abs(x_recon - x), dim=[1, 2])  # shape: (batch,)
        return x_mean, x_logvar, x_recon, z_mean, z_logvar, attn_weights, focus

    def phase1(self, x):
        _, _, x_recon, _, _, attn_weights, focus = self.forward(x)
        return x_recon, attn_weights, focus

    def phase2(self, x):
        return self.phase1(x)
