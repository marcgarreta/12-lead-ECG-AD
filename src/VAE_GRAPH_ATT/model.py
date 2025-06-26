import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding as in "Attention is All You Need".
    """
    def __init__(self, d_model, max_len=500):
        super().__init__()
        # Create constant positional encoding matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return self.pe[:, :seq_len]

class GaussianNoise(nn.Module):
    def __init__(self, std=0.01):
        super().__init__()
        self.std = std
    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x

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

class MA(nn.Module):
    def __init__(self, n_leads, latent_dim, n_heads=8, max_len=500):
        super().__init__()
        self.to_q = nn.Linear(n_leads, latent_dim)
        self.pos_enc = PositionalEncoding(latent_dim, max_len=max_len)
        self.attn = nn.MultiheadAttention(latent_dim, num_heads=n_heads, batch_first=True)

    def forward(self, x, z):
        # x: (batch, seq_len, n_leads)
        # z: (batch, seq_len, latent_dim)
        # Project inputs to queries
        q = self.to_q(x)  # shape (batch, seq_len, latent_dim)
        # Add positional encodings
        pe_q = self.pos_enc(q)
        pe_z = self.pos_enc(z)
        q = q + pe_q
        z = z + pe_z
        # Apply multi-head attention: query, key, value
        attn_out, attn_weights = self.attn(q, z, z, need_weights=True)
        return attn_out, attn_weights

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
        # Use softplus to ensure positive variance and avoid numerical issues
        raw_logvar = self.linear_logvar(out)
        x_var = F.softplus(raw_logvar) + 1e-6
        x_logvar = torch.log(x_var)
        sigma = torch.sqrt(x_var)
        eps = torch.randn_like(x_mean)
        x_recon = x_mean + sigma * eps
        return x_mean, x_logvar, x_recon


class MA_VAE(nn.Module):
    def __init__(self, seq_len, n_leads, latent_dim):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = VAEEncoder(seq_len, n_leads, latent_dim)
        self.ma = MA(n_leads, latent_dim, max_len=seq_len)
        self.decoder = VAEDecoder(seq_len, n_leads, latent_dim)
        # Spatial attention over leads: sequence length = n_leads, embed_dim = seq_len
        self.spatial_attn = MultiheadAttention(embed_dim=seq_len, num_heads=4, batch_first=True)

    def forward(self, x):
        # save original for focus; permute for encoder/attention
        x_orig = x
        # Encode
        z_mean, z_logvar, z, _ = self.encoder(x)
        # Spatial attention over leads at each time step
        # x: (batch, seq_len, n_leads)
        x_sp = x.permute(0, 2, 1)  # (batch, n_leads, seq_len)
        # attend across leads
        sp_out, sp_w = self.spatial_attn(x_sp, x_sp, x_sp, need_weights=True, average_attn_weights=False)
        # sp_out: (batch, n_leads, seq_len) → transpose back
        x2 = sp_out.permute(0, 2, 1)  # (batch, seq_len, n_leads)
        # use x2 for temporal attention
        attn_out, attn_weights = self.ma(x2, z)
        # Decode
        x_mean, x_logvar, x_recon = self.decoder(attn_out)
        # compute focus against original signal
        focus = torch.mean(torch.abs(x_recon - x_orig), dim=[1, 2])
        return x_mean, x_logvar, x_recon, z_mean, z_logvar, sp_w, attn_weights, focus

    def phase1(self, x):
        _, _, x_recon, _, _, sp_w, attn_weights, focus = self.forward(x)
        return x_recon, sp_w, attn_weights, focus

    def phase2(self, x):
        return self.phase1(x)
