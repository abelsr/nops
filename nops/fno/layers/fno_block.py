from typing import Literal, List

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as _spectral_norm

from .mlp import MLP
from .ffn import FeedForwardNet
from .spectral_convolution import SpectralConvolution, NativeSpectralConv


class FourierBlock(nn.Module):
    """
        Fourier block with optional residual connections and 
        spectral normalization.
        
        Architecture:
        1. SpectralConvolution (with optional spectral norm)
        2. MLP (1x1 conv)
        3. Local convolution (3x3)
        4. Skip connection to input
        
        With residual=True and in_channels==out_channels::
          x_out = x_in + activation(sum_of_components)  # identity skip
    """
    def __init__(
        self, 
        modes: List[int], 
        in_channels: int, 
        out_channels: int, 
        hidden_size: int | None = None, 
        activation: nn.Module = nn.GELU(), 
        mid_net_type: Literal['mlp', 'ffn'] = 'mlp',
        bias: bool = False,
        spectral_norm: bool = False,
        residual: bool = False,
        native_spectral_conv: bool = False,
    ) -> None:
        """        
        Parameters:
        -----------
        modes: List[int] or int (Required)
            Number of Fourier modes to use in the Fourier layer (SpectralConvolution).
        in_channels: int (Required)
            Number of input channels.
        out_channels: int (Required)
            Number of output channels.
        hidden_size: int (Optional)
            Number of hidden units in the MLP layer.
        activation: nn.Module (Optional)
            Activation function. Default: nn.GELU().
        mid_net_type: str (Optional)
            Type of intermediate network: 'mlp' or 'ffn'.
        spectral_norm: bool (Optional)
            Apply spectral normalization to SpectralConv weights. Default: False.
        residual: bool (Optional)
            Use identity skip connection when in_channels == out_channels. Default: False.
        native_spectral_conv: bool (Optional)
            Use NativeSpectralConv (complex64, Li et al. 2020) instead of the
            legacy real/imag split implementation. Faster + AMP-safe. Default: False.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.activation = activation
        self.modes = modes
        self.dim = len(self.modes)
        self.bias = bias
        self.residual = residual and (in_channels == out_channels)

        # Fourier layer — native (E05) or legacy
        if native_spectral_conv:
            self.fourier = NativeSpectralConv(in_channels, out_channels, modes)
        else:
            self.fourier = SpectralConvolution(in_channels, out_channels, modes, factorization='dense')
        # Spectral norm not yet supported for SpectralConv (no 'weight' param)
        
        # MLP layer
        if self.hidden_size is not None:
            if mid_net_type == 'mlp':
                self.mlp = MLP(len(self.modes), in_channels, out_channels, self.hidden_size, activation)
            elif mid_net_type == 'ffn':
                self.mlp = FeedForwardNet(
                    input_dim=in_channels,
                    hidden1=self.hidden_size,
                    hidden2=self.hidden_size,
                    output_dim=out_channels,
                    dropout=0.0,
                    non_linearity='gelu'
                )
            else:
                raise NotImplementedError(f"Mid network type '{mid_net_type}' is not implemented.")
        
        # Local convolution layer
        if self.dim == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        elif self.dim == 3:
            self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        
        # Skip projection if in_channels != out_channels for residual mode
        if self.residual and in_channels != out_channels:
            if self.dim == 2:
                self.skip_proj = nn.Conv2d(in_channels, out_channels, 1)
            elif self.dim == 3:
                self.skip_proj = nn.Conv3d(in_channels, out_channels, 1)
            else:
                self.skip_proj = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip_proj = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        ----------
        x: torch.Tensor
            Input tensor of shape [batch, channels, *sizes]
        
        Returns:
        -------
        x: torch.Tensor
            Output tensor of shape [batch, channels, *sizes]
        """
        assert x.size(1) == self.in_channels, f"Input channels must be {self.in_channels} but got {x.size(1)}"
        original_size = x.size()
        
        # Save skip connection (possibly projected)
        skip = x
        if self.skip_proj is not None:
            skip = self.skip_proj(skip)
        
        # Fourier layer
        x_ft = self.fourier(x)
        
        # MLP layer (1x1 conv)
        if self.hidden_size is not None:
            x_mlp = self.mlp(x)
        
        # Local convolution (3x3)
        if self.dim == 2 or self.dim == 3:
            x_conv = self.conv(x)
        else:
            x_conv = self.conv(x.reshape(original_size[0], self.in_channels, -1)).reshape(*original_size)
        
        # Sum all components
        out = x_ft + x_conv
        if self.hidden_size is not None:
            out = out + x_mlp
        
        # Residual connection or standard activation
        if self.residual:
            out = skip + self.activation(out)
        else:
            out = self.activation(out)
            
        return out
