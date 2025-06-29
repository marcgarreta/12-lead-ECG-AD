
import torch
import torch.nn as nn
from torch.nn import functional as F

# Adaptation of CAE-M Model (TKDE 21)
class CAE_M(nn.Module):
    def __init__(self, in_channels=12):
        super(CAE_M, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),  # [B, 32, L/2]
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),           # [B, 64, L/4]
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),          # [B, 128, L/8]
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, in_channels, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded[:, :, :x.shape[2]] 
        return decoded
