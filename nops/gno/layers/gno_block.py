"""
Graph Neural Operator Block.
Implements a single GNO layer with integral transform and optional skip connection.
"""

import torch
import torch.nn as nn
from typing import Optional
from .integral_transform import IntegralTransform


class GNOBlock(nn.Module):
    """
    Graph Neural Operator Block.

    Applies integral transform followed by optional skip connection and activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        hidden_channels (int): Number of hidden channels in the kernel network.
        n_layers (int): Number of layers in the kernel network. Default: 1.
        activation (nn.Module): Activation function. Default: nn.GELU().
        skip (str): Type of skip connection ('skip', 'proj', or None). Default: 'skip'.
        eps (float): Small value for numerical stability. Default: 1e-12.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 1,
        activation: nn.Module = nn.GELU(),
        skip: str = "skip",
        eps: float = 1e-12,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip = skip
        self.eps = eps

        # Integral transform (kernel network)
        self.integral_transform = IntegralTransform(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            activation=activation,
        )

        # Skip connection
        if skip == "skip" and in_channels != out_channels:
            self.skip_proj = nn.Linear(in_channels, out_channels)
        elif skip == "proj":
            self.skip_proj = nn.Linear(in_channels, out_channels)
        else:
            self.skip_proj = None

        self.skip = skip if skip is not None else None

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the GNO block.

        Args:
            x (torch.Tensor): Input features [batch, n_points, in_channels]
            pos (torch.Tensor): Node positions [batch, n_points, pos_dim] or [n_points, pos_dim]
            edge_index (torch.Tensor): Edge indices [2, n_edges]
            edge_weights (torch.Tensor, optional): Edge weights [n_edges]

        Returns:
            torch.Tensor: Output features [batch, n_points, out_channels]
        """
        # Apply integral transform
        out = self.integral_transform(x, pos, edge_index, edge_weights)

        # Apply skip connection
        if self.skip is not None:
            if self.skip_proj is not None:
                skip = self.skip_proj(x)
            else:
                skip = x
            out = out + skip

        return out
