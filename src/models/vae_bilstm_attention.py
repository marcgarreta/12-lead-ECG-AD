import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

# Inspired by OmniAnomaly (KDD ’19) and MA-VAE (Multi-head Attention-based VAE; arXiv:2309.02253)
class VAE(nn.Module):
    def __init__(self,
                 n_leads: int,
                 n_hidden_enc: tuple = (512, 256),
                 n_hidden_dec: tuple = (256, 512),
                 n_latent: int = 64,
                 beta: float = 0.01):
        super().__init__()
        self.beta = beta
        self.n_feats = n_leads
        self.n_latent = n_latent

        # Encoder
        self.noise = nn.Dropout(p=0.01)              # GaussianNoise 
        self.enc_lstm1 = nn.LSTM(input_size=n_leads,
                                 hidden_size=n_hidden_enc[0],
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=True)

        self.enc_lstm2 = nn.LSTM(input_size=2 * n_hidden_enc[0],
                                 hidden_size=n_hidden_enc[1],
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=True)

        enc_out_dim = 2 * n_hidden_enc[1]           # because bidirectional

        self.to_stats = nn.Sequential(
            nn.Linear(enc_out_dim, enc_out_dim // 2),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(enc_out_dim // 2, 2 * n_latent)
        )

        # Multihead Attention
        self.qk_proj = nn.Linear(n_leads, n_latent)  # project input to latent dim
        self.ma = MultiheadAttention(embed_dim=n_latent,
                                     num_heads=8,
                                     batch_first=True)

        # Lead-wise attention
        self.lead_embed = nn.Linear(1, self.n_latent)
        self.lead_attn = MultiheadAttention(embed_dim=self.n_latent,
                                            num_heads=4,
                                            batch_first=True)

        # Decoder
        self.dec_lstm1 = nn.LSTM(input_size=n_latent,
                                 hidden_size=n_hidden_dec[0],
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=True)

        self.dec_lstm2 = nn.LSTM(input_size=2 * n_hidden_dec[0],
                                 hidden_size=n_hidden_dec[1],
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=True)

        dec_out_dim = 2 * n_hidden_dec[1]
        self.to_recon = nn.Sequential(
            nn.Linear(dec_out_dim, dec_out_dim // 2),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(dec_out_dim // 2, 2 * n_leads)
        )

    def forward(self, x):
        """
        x: (B, T, n_leads)
        Returns mean & logvar of reconstruction plus latent stats.
        lead_w: (B, T, n_leads) – averaged attention weights over leads for each time step.
        """
        x_noisy = self.noise(x)

        # Encoder
        h, _ = self.enc_lstm1(x_noisy)
        h, _ = self.enc_lstm2(h)

        # per‑timestep latent parameters
        stats = self.to_stats(h)                 # (B, T, 2*latent)
        mu, logvar = stats.chunk(2, dim=-1)

        # reparameterisation
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)  # (B,T,latent)

        # Lead-wise attention
        B, T, L = x.shape  # L == self.n_feats
        x_leads = x.reshape(B * T, L, 1)           # (B*T, L, 1)
        lead_emb = self.lead_embed(x_leads)        # (B*T, L, latent)
        lead_out, lead_w = self.lead_attn(lead_emb, lead_emb, lead_emb,
                                          need_weights=True)      # lead_w: (B*T, heads, L, L)

        lead_w = lead_w.mean(dim=1)                # (B*T, L)
        lead_w = lead_w.view(B, T, L)              # (B, T, L)

        lead_out = lead_out.mean(dim=1)            # (B*T, latent)
        lead_out = lead_out.view(B, T, self.n_latent)
        z = z + lead_out

        # Multihead Attention
        qk = self.qk_proj(x)                     # (B,T,latent)
        A, A_weights = self.ma(qk, qk, z, need_weights=True)               # (B,T,latent)

        # Decoder
        d, _ = self.dec_lstm1(A)
        d, _ = self.dec_lstm2(d)
        rec_stats = self.to_recon(d)            # (B,T,2*n_leads)
        x_mean, x_logvar = rec_stats.chunk(2, dim=-1)

        return x_mean, x_logvar, mu, logvar, A_weights, lead_w

    # ---------- loss ----------
    def loss_function(self, x, x_mean, x_logvar, mu, logvar):
        """
        Loss function for the VAE: reconstruction loss + KL divergence.
        """
        x_logvar = torch.clamp(x_logvar, -10.0, 10.0)
        recon = -torch.distributions.Normal(
            loc=x_mean, scale=torch.exp(0.5 * x_logvar)).log_prob(x)
        recon_loss = recon.sum(dim=[1, 2]).mean()

        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl.mean()

        total = recon_loss + self.beta * kl_loss
        return total, recon_loss, kl_loss
