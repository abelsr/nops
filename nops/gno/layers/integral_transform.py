"""
Integral Transform layer for Graph Neural Operators.
Implements the learnable kernel integral that is the core of GNO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class IntegralTransform(nn.Module):
    """
    Integral Transform layer for Graph Neural Operators.

    This layer implements the kernel integration operation:
    (Kv)(x) = ∫_Ω κ(x, y) v(y) dy

    where κ is a learnable kernel function parameterized by a neural network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        hidden_channels (int): Number of hidden channels in the kernel network.
        n_layers (int): Number of layers in the kernel network. Default: 1.
        activation (nn.Module): Activation function to use. Default: F.gelu.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 1,
        activation: callable = F.gelu,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.activation = activation

        # Build the kernel network (MLP)
        # Takes relative position as input, outputs kernel weights
        self.layers = nn.ModuleList()

        # Input dimension is 2 (for 2D relative position) or 3 (for 3D)
        # We'll determine this dynamically in forward pass
        self.input_dim = None

        # First layer (will be initialized properly in first forward pass)
        self.first_layer = None

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(max(0, n_layers - 1)):
            self.hidden_layers.append(nn.Linear(hidden_channels, hidden_channels))

        # Output layer
        self.output_layer = nn.Linear(hidden_channels, in_channels * out_channels)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the integral transform.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, n_points, in_channels]
            pos (torch.Tensor): Position tensor of shape [batch, n_points, pos_dim]
                               or [n_points, pos_dim]
            edge_index (torch.Tensor): Graph connectivity of shape [2, n_edges]
                                       containing global node indices
            edge_weights (torch.Tensor, optional): Edge weights of shape [n_edges]
                                                  for weighted aggregation

        Returns:
            torch.Tensor: Output tensor of shape [batch, n_points, out_channels]
        """
        # Handle batch dimension
        if pos.dim() == 2:  # [n_points, pos_dim]
            n_points = pos.shape[0]
            pos_dim = pos.shape[1]
            batch_size = x.shape[0]
            # Expand pos to batch dimension
            pos = pos.unsqueeze(0).expand(batch_size, -1, -1)
        elif pos.dim() == 3:  # [batch, n_points, pos_dim]
            batch_size, n_points, pos_dim = pos.shape
        else:
            raise ValueError(f"Position tensor must be 2D or 3D, got {pos.dim()}D")

        # Initialize kernel network layers if not done yet
        if self.input_dim is None:
            # Input to kernel is concatenated positions: [pos_i, pos_j]
            self.input_dim = 2 * pos_dim
            # Create first layer with correct dimensions
            self.first_layer = nn.Linear(self.input_dim, self.hidden_channels)
            # Move to same device as other parameters
            self.first_layer.to(next(self.parameters()).device)

        # Flatten positions and input for batch processing
        # pos_flat: [batch * n_points, pos_dim]
        pos_flat = pos.view(-1, pos_dim)

        # x_flat: [batch * n_points, in_channels]
        x_flat = x.view(-1, self.in_channels)

        # Extract source and target indices from edge_index
        src_idx = edge_index[0]  # Source nodes (global indices)
        dst_idx = edge_index[1]  # Target nodes (global indices)

        # Get positions for source and target nodes using flattened positions
        pos_src = pos_flat[src_idx]  # [n_edges, pos_dim]
        pos_dst = pos_flat[dst_idx]  # [n_edges, pos_dim]

        # Concatenate positions for kernel input: [pos_i, pos_j]
        # Shape: [n_edges, 2*pos_dim]
        pos_enc = torch.cat([pos_src, pos_dst], dim=-1)

        # Pass through kernel network
        # Shape: [n_edges, 2*pos_dim]

        # Apply first layer
        h = self.activation(self.first_layer(pos_enc))

        # Apply hidden layers
        for layer in self.hidden_layers:
            h = self.activation(layer(h))

        # Output layer: [n_edges, in_channels * out_channels]
        h = self.output_layer(h)

        # Reshape kernel weights: [n_edges, in_channels, out_channels]
        kernel_weights = h.view(-1, self.in_channels, self.out_channels)

        # Get features for source nodes: [n_edges, in_channels]
        x_src = x_flat[src_idx]

        # Apply kernel transformation: κ(x_i, x_j) * x_j
        # kernel_weights: [n_edges, in_channels, out_channels]
        # x_src: [n_edges, in_channels, 1]
        x_src_expanded = x_src.unsqueeze(-1)

        # Element-wise multiplication and sum over in_channels
        # Shape: [n_edges, out_channels]
        msg = torch.sum(kernel_weights * x_src_expanded, dim=1)

        # Apply edge weights if provided
        if edge_weights is not None:
            # edge_weights: [n_edges]
            msg = msg * edge_weights.unsqueeze(-1)

        # Aggregate messages for each destination node
        # Output: [batch * n_points, out_channels]
        out_flat = torch.zeros(
            batch_size * n_points, self.out_channels, device=x.device, dtype=x.dtype
        )

        # Use index_add to accumulate messages to destination nodes
        out_flat.index_add_(0, dst_idx, msg)

        # Reshape output: [batch, n_points, out_channels]
        out = out_flat.view(batch_size, n_points, self.out_channels)

        return out
