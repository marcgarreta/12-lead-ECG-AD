import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

class VAE(nn.Module):
    def __init__(self, n_leads: int, n_hidden: int = 64, n_latent: int = 8, n_layers: int = 2, beta: float = 0.01):
        super(VAE, self).__init__()
        self.name = 'VAE'
        self.lr = 0.002
        self.beta = beta
        self.n_feats = n_leads
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        # GRU with batch_first to handle windows
        self.gru = nn.GRU(
            input_size=self.n_feats,
            hidden_size=self.n_hidden,
            num_layers=n_layers,
            batch_first=True
        )
        # per-timestep encoder to mu/logvar, with Dropout after each PReLU
        self.encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.PReLU(),
            #nn.Dropout(p=0.1),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.PReLU(),
            #nn.Dropout(p=0.1),
            nn.Linear(self.n_hidden, 2 * self.n_latent)
        )
        # decoder maps latent back to reconstruction stats (mean + logvar), with Dropout after first PReLU
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden),
            nn.PReLU(),
            #nn.Dropout(p=0.1),
            nn.Linear(self.n_hidden, 2 * self.n_feats)
        )
        # small self-attention layer over time dimension
        self.attn = MultiheadAttention(embed_dim=self.n_hidden, num_heads=4, batch_first=True)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (B, T, n_leads)
        Returns:
            x_mean: (B, T, n_leads)
            x_logvar: (B, T, n_leads)
            mu:       (B, T, n_latent)
            logvar:   (B, T, n_latent)
        """
        # GRU encoding
        h_seq, _ = self.gru(x)  # (B, T, n_hidden)

        # apply self-attention to GRU outputs
        # h_seq shape: (B, T, n_hidden)
        attn_out, _ = self.attn(h_seq, h_seq, h_seq)
        h_seq = attn_out  # use attention-enhanced features

        # flatten time dimension for vectorized per-step VAE
        B, T, _ = h_seq.size()
        h_flat = h_seq.reshape(B * T, self.n_hidden)  # (B*T, n_hidden)

        # encode to stats
        stats = self.encoder(h_flat)                # (B*T, 2*n_latent)
        mu, logvar = stats.chunk(2, dim=-1)         # each (B*T, n_latent)

        # reparameterize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std                          # (B*T, n_latent)

        # decode
        x_stats = self.decoder(z)                   # (B*T, 2*n_feats)
        x_mean, x_logvar = x_stats.chunk(2, dim=-1) # each (B*T, n_feats)
        # reshape back to sequence
        x_recon = x_mean.reshape(B, T, self.n_feats)
        x_logvar = x_logvar.reshape(B, T, self.n_feats)
        mu = mu.reshape(B, T, self.n_latent)
        logvar = logvar.reshape(B, T, self.n_latent)

        return x_recon, x_logvar, mu, logvar

    def loss_function(self, x: torch.Tensor, x_mean: torch.Tensor, x_logvar: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Compute VAE loss over batch and time.
        """
        min_logvar, max_logvar = -10.0, 10.0
        x_logvar = torch.clamp(x_logvar, min=min_logvar, max=max_logvar)
        
        dist = torch.distributions.Normal(x_mean, torch.exp(0.5 * x_logvar))
        log_px = dist.log_prob(x).sum(dim=[1,2]).mean()  # sum over T and lead dims, mean over batch
        recon_loss = -log_px

        # KL per time-step and latent dim, then mean
        kl_per = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, T, n_latent)
        kl_loss = kl_per.mean()
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss
