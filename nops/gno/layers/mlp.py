"""
Multi-Layer Perceptron layer for GNO components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class MLP(nn.Module):
    """
    Multi-Layer Perceptron.

    Args:
        layers (List[int]): List of layer sizes.
        activation (nn.Module): Activation function. Default: nn.GELU().
        dropout (float): Dropout rate. Default: 0.0.
        batch_norm (bool): Whether to use batch normalization. Default: False.
    """

    def __init__(
        self,
        layers: List[int],
        activation: nn.Module = nn.GELU(),
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()

        self.layers = layers
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm

        # Build network
        self.net = nn.ModuleList()

        for i in range(len(layers) - 1):
            self.net.append(nn.Linear(layers[i], layers[i + 1]))

            # Add activation, batch norm, and dropout (except after last layer)
            if i < len(layers) - 2:
                if batch_norm:
                    self.net.append(nn.BatchNorm1d(layers[i + 1]))
                self.net.append(activation)
                if dropout > 0:
                    self.net.append(nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, *, in_features]

        Returns:
            torch.Tensor: Output tensor of shape [batch, *, out_features]
        """
        # Flatten all dimensions except last
        input_shape = x.shape
        flat_x = x.view(-1, input_shape[-1])

        # Apply network
        for layer in self.net:
            flat_x = layer(flat_x)

        # Restore original shape
        output_shape = list(input_shape[:-1]) + [flat_x.shape[-1]]
        return flat_x.view(output_shape)
