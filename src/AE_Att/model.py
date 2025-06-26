import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return self.pe[:, : x.size(1)]

class Encoder(nn.Module):
    def __init__(self, n_leads, latent_dim):
        super().__init__()
        self.lstm1 = nn.LSTM(n_leads, 512,  batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(512*2, 256,   batch_first=True, bidirectional=True)
        self.proj   = nn.Linear(256*2, latent_dim)
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return self.proj(x)                   # (B, W, latent_dim)

class TemporalAttention(nn.Module):
    def __init__(self, n_leads, latent_dim, n_heads=8):
        super().__init__()
        self.to_q    = nn.Linear(n_leads, latent_dim)
        self.pos_enc = PositionalEncoding(latent_dim)
        self.attn    = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
    def forward(self, x, z):
        # x: (B, W, 12), z: (B, W, latent_dim)
        q = self.to_q(x) + self.pos_enc(self.to_q(x))
        kv = z + self.pos_enc(z)
        out, weights = self.attn(q, kv, kv, need_weights=True)
        return out, weights

class Decoder(nn.Module):
    def __init__(self, latent_dim, n_leads):
        super().__init__()
        self.lstm1 = nn.LSTM(latent_dim, 256,  batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256*2,     512,  batch_first=True, bidirectional=True)
        self.proj   = nn.Linear(512*2, n_leads)
    def forward(self, z):
        z, _ = self.lstm1(z)
        z, _ = self.lstm2(z)
        return self.proj(z)                   # (B, W, 12)

class AEWithAttention(nn.Module):
    def __init__(self, seq_len, n_leads, latent_dim, n_heads=8):
        super().__init__()
        self.encoder = Encoder(n_leads, latent_dim)
        self.attn    = TemporalAttention(n_leads, latent_dim, n_heads)
        self.decoder = Decoder(latent_dim, n_leads)

    def forward(self, x):
        """
        x: (batch, seq_len, n_leads)
        returns:
          recon:    (batch, seq_len, n_leads)
          attn_w:   (batch, heads, seq_len, seq_len)
          recon_err:(batch, seq_len, n_leads)  # per-timestep error if you like
        """
        z     = self.encoder(x)
        z_attn, w = self.attn(x, z)
        recon    = self.decoder(z_attn)
        return recon, w
