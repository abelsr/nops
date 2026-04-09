"""
Mollified Graph Neural Operator (mGNO) implementation.

mGNO extends GNO with mollification to enable automatic differentiation
on arbitrary geometries. This is the first method to leverage automatic
differentiation and compute exact gradients on arbitrary geometries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from nops.gno.layers.gno_block import GNOBlock
from nops.gno.layers.neighbor_search import knn_graph


class MollifiedKernel(nn.Module):
    """
    Mollified kernel for GNO.

    Uses a mollification function to make the kernel smooth and
    differentiable. The mollified kernel is:
    κ_m(x, y) = κ(x, y) * w(||x - y||)

    where w is a weighting function (e.g., half-cosine, polynomial).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 1,
        activation: callable = F.gelu,
        weight_type: str = "half_cos",
        radius: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.weight_type = weight_type
        self.radius = radius

        # Kernel network
        self.layers = nn.ModuleList()
        input_dim = 1  # Normalized distance: r / R

        self.layers.append(nn.Linear(input_dim, hidden_channels))
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(nn.Linear(hidden_channels, in_channels * out_channels))

        self.activation = activation

    def _weight_function(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Compute mollification weights based on distance.

        Args:
            dist (torch.Tensor): Distance tensor

        Returns:
            torch.Tensor: Weight tensor of same shape
        """
        if self.weight_type == "half_cos":
            # Half-cosine: w(r) = cos(π * r / (2R)) for r <= R, else 0
            r = dist / self.radius
            weights = torch.where(
                r <= 1, torch.cos(torch.pi * r / 2), torch.zeros_like(r)
            )
        elif self.weight_type == "poly":
            # Polynomial: w(r) = (1 - r/R)^2 for r <= R
            r = dist / self.radius
            weights = torch.where(r <= 1, (1 - r) ** 2, torch.zeros_like(r))
        elif self.weight_type == "gaussian":
            # Gaussian: w(r) = exp(-r^2 / (2R^2))
            weights = torch.exp(-(dist**2) / (2 * self.radius**2))
        else:
            weights = torch.ones_like(dist)

        return weights

    def forward(self, pos_src: torch.Tensor, pos_dst: torch.Tensor) -> torch.Tensor:
        """
        Compute mollified kernel weights.

        Args:
            pos_src (torch.Tensor): Source positions [n_edges, pos_dim]
            pos_dst (torch.Tensor): Target positions [n_edges, pos_dim]

        Returns:
            torch.Tensor: Kernel weights [n_edges, in_channels, out_channels]
        """
        # Compute relative distances
        rel_pos = pos_dst - pos_src
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # [n_edges, 1]

        # Compute mollification weights
        weights = self._weight_function(dist.squeeze(-1))  # [n_edges]

        # Pass through kernel network
        # Input: normalized distance
        h = dist / self.radius  # Normalize by radius
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)

        # Reshape to kernel weights
        kernel = h.view(-1, self.in_channels, self.out_channels)

        # Apply mollification weights
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # [n_edges, 1, 1]
        kernel = kernel * weights

        return kernel


class mGNOBlock(nn.Module):
    """
    Mollified GNO Block.

    Uses a mollified kernel that is fully differentiable, enabling
    automatic differentiation on arbitrary geometries.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 1,
        activation: nn.Module = nn.GELU(),
        weight_type: str = "half_cos",
        radius: float = 0.1,
        skip: str = "skip",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius

        # Mollified kernel
        self.kernel = MollifiedKernel(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            activation=activation,
            weight_type=weight_type,
            radius=radius,
        )

        # Skip connection
        if skip == "skip" and in_channels != out_channels:
            self.skip_proj = nn.Linear(in_channels, out_channels)
        else:
            self.skip_proj = None

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of mGNO block.

        Args:
            x (torch.Tensor): Input features [batch, n_points, in_channels]
            pos (torch.Tensor): Node positions [batch, n_points, pos_dim]
            edge_index (torch.Tensor): Edge indices [2, n_edges]
            edge_weights (torch.Tensor, optional): Edge weights [n_edges]

        Returns:
            torch.Tensor: Output features [batch, n_points, out_channels]
        """
        batch_size, n_points, _ = x.shape

        # Flatten for processing
        x_flat = x.view(-1, self.in_channels)
        pos_flat = pos.view(-1, pos.shape[-1])

        # Get edge endpoints
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        pos_src = pos_flat[src_idx]
        pos_dst = pos_flat[dst_idx]

        # Compute mollified kernel
        kernel = self.kernel(pos_src, pos_dst)  # [n_edges, in_channels, out_channels]

        # Get source features
        x_src = x_flat[src_idx]  # [n_edges, in_channels]

        # Apply kernel: κ(x_i, x_j) * x_j
        x_src_expanded = x_src.unsqueeze(-1)  # [n_edges, in_channels, 1]
        msg = torch.sum(kernel * x_src_expanded, dim=1)  # [n_edges, out_channels]

        # Apply edge weights
        if edge_weights is not None:
            msg = msg * edge_weights.unsqueeze(-1)

        # Aggregate to destination nodes
        out_flat = torch.zeros(
            batch_size * n_points, self.out_channels, device=x.device, dtype=x.dtype
        )
        out_flat.index_add_(0, dst_idx, msg)

        # Reshape and apply skip
        out = out_flat.view(batch_size, n_points, self.out_channels)

        if self.skip_proj is not None:
            out = out + self.skip_proj(x)

        return out


class mGNO(nn.Module):
    """
    Mollified Graph Neural Operator (mGNO).

    Extends GNO with mollification for fully differentiable operations
    on arbitrary geometries. Enables automatic differentiation for
    physics-informed losses.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_layers (int): Number of mGNO layers.
        hidden_channels (int): Number of hidden channels.
        lifting_channels (int, optional): Channels for lifting layer.
        projection_channels (int, optional): Channels for projection layer.
        activation (nn.Module): Activation function.
        radius (float): Radius for mollification.
        weight_type (str): Type of weight function ('half_cos', 'poly', 'gaussian').
        add_grid (bool): Whether to append grid coordinates.
        skip (str): Type of skip connection.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 4,
        hidden_channels: int = 64,
        lifting_channels: Optional[int] = None,
        projection_channels: Optional[int] = None,
        activation: nn.Module = nn.GELU(),
        radius: float = 0.1,
        weight_type: str = "half_cos",
        add_grid: bool = True,
        skip: str = "skip",
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.add_grid = add_grid

        effective_in = in_channels + (2 if add_grid else 0)

        # Lifting
        if lifting_channels:
            self.lift = nn.Sequential(
                nn.Linear(effective_in, lifting_channels),
                activation,
                nn.Linear(lifting_channels, hidden_channels),
            )
        else:
            self.lift = nn.Linear(effective_in, hidden_channels)

        # mGNO layers
        self.layers = nn.ModuleList(
            [
                mGNOBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    hidden_channels=hidden_channels,
                    n_layers=1,
                    activation=activation,
                    weight_type=weight_type,
                    radius=radius,
                    skip=skip,
                )
                for _ in range(num_layers)
            ]
        )

        # Projection
        if projection_channels:
            self.proj = nn.Sequential(
                nn.Linear(hidden_channels, projection_channels),
                activation,
                nn.Linear(projection_channels, out_channels),
            )
        else:
            self.proj = nn.Linear(hidden_channels, out_channels)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input [batch, channels, n_points]
            pos (torch.Tensor): Positions [batch, n_points, 2]
            edge_index (torch.Tensor, optional): Pre-computed edges
            edge_weights (torch.Tensor, optional): Edge weights

        Returns:
            torch.Tensor: Output [batch, out_channels, n_points]
        """
        # Handle input format
        if x.shape[1] == self.in_channels:
            x = x.permute(0, 2, 1)

        batch_size, n_points, _ = x.shape

        # Add grid if requested
        if self.add_grid:
            x = torch.cat([x, pos], dim=-1)

        # Lift
        x = self.lift(x)

        # Build graph if not provided
        if edge_index is None:
            edge_index = knn_graph(pos, k=8)

        edge_index = edge_index.to(x.device)
        if edge_weights is not None:
            edge_weights = edge_weights.to(x.device)

        # mGNO layers
        for layer in self.layers:
            x = layer(x, pos, edge_index, edge_weights)
            if self.dropout:
                x = self.dropout(x)

        # Project
        x = self.proj(x)

        return x.permute(0, 2, 1)


if __name__ == "__main__":
    # Test mGNO
    model = mGNO(
        in_channels=3,
        out_channels=1,
        num_layers=2,
        hidden_channels=32,
        radius=0.15,
        weight_type="half_cos",
        add_grid=True,
    )

    batch_size = 2
    n_points = 25
    x = torch.randn(batch_size, 3, n_points)
    pos = torch.rand(batch_size, n_points, 2)

    output = model(x, pos=pos)
    print(f"mGNO Input: {x.shape}, Output: {output.shape}")
    print(
        "mGNO test passed!" if output.shape == (batch_size, 1, n_points) else "Failed!"
    )
