import torch
import torch.nn as nn
from typing import List, Optional
from .spectral_convolution import SpectralConvolution
from .mlp import MLP

class SequentialFourierBlock(nn.Module):
    """
    FNO block as shown in the architecture diagram:
    Spectral Convolution -> MLP -> Convolution
    """
    def __init__(
        self, 
        modes: List[int], 
        in_channels: int, 
        out_channels: int, 
        hidden_size: int, 
        activation: nn.Module = nn.GELU()
    ):
        super().__init__()
        self.modes = modes
        self.dim = len(modes)
        
        # 1. Spectral Convolution
        self.fourier = SpectralConvolution(in_channels, out_channels, modes, factorization='dense')
        
        # 2. MLP (Point-wise)
        self.mlp = MLP(self.dim, out_channels, out_channels, hidden_size, activation)
        
        # 3. Convolution
        if self.dim == 1:
            self.conv = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        elif self.dim == 2:
            self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        elif self.dim == 3:
            self.conv = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spectral Convolution
        x = self.fourier(x)
        # MLP
        x = self.mlp(x)
        # Convolution
        x = self.conv(x)
        return x
