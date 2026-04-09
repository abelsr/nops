"""
Graph Neural Operator (GNO) model for learning operators on irregular meshes.
"""

from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nops.gno.layers.gno_block import GNOBlock
from nops.gno.layers.mlp import MLP
from nops.gno.layers.neighbor_search import radius_graph, knn_graph


class GNO(nn.Module):
    """
    Graph Neural Operator (GNO) model.

    GNO implements kernel integration with graph structures and is applicable
    to complex geometries and irregular grids. It defines the graph connection
    in a ball defined on continuous physical space, making it discretization-convergent.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_gno_layers (int): Number of GNO layers in the model.
        hidden_channels (int): Number of hidden channels.
        lifting_channels (int, optional): Number of channels in the lifting layer.
        projection_channels (int, optional): Number of channels in the projection layer.
        activation (nn.Module): Activation function. Default: nn.GELU().
        radius (float, optional): Radius for neighbor search. Default: 0.1.
        k_neighbors (int, optional): Number of neighbors for KNN graph.
                                     If None, uses radius-based graph.
        neighbor_strategy (str): Strategy for building graph ('radius' or 'knn').
        add_grid (bool): Whether to append grid coordinates to input.
        skip (str): Type of skip connection ('skip', 'proj', or None).
        dropout (float): Dropout rate. Default: 0.0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_gno_layers: int = 4,
        hidden_channels: int = 64,
        lifting_channels: Optional[int] = None,
        projection_channels: Optional[int] = None,
        activation: nn.Module = nn.GELU(),
        radius: float = 0.1,
        k_neighbors: Optional[int] = None,
        neighbor_strategy: str = "radius",
        add_grid: bool = True,
        skip: str = "skip",
        dropout: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_gno_layers = num_gno_layers
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.activation = activation
        self.radius = radius
        self.k_neighbors = k_neighbors
        self.neighbor_strategy = neighbor_strategy
        self.add_grid = add_grid
        self.skip = skip
        self.dropout = dropout

        # Determine effective input channels
        effective_in_channels = in_channels + (
            2 if add_grid else 0
        )  # pos_dim=2 for now

        # Lifting layer
        if lifting_channels is not None:
            self.lift = nn.Sequential(
                nn.Linear(effective_in_channels, lifting_channels),
                activation,
                nn.Linear(lifting_channels, hidden_channels),
            )
            self.lift_channels = lifting_channels
        else:
            self.lift = nn.Linear(effective_in_channels, hidden_channels)
            self.lift_channels = None

        # GNO layers
        self.gno_layers = nn.ModuleList(
            [
                GNOBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    hidden_channels=hidden_channels,
                    n_layers=1,
                    activation=activation,
                    skip=skip,
                )
                for _ in range(num_gno_layers)
            ]
        )

        # Projection layer
        if projection_channels is not None:
            self.proj = nn.Sequential(
                nn.Linear(hidden_channels, projection_channels),
                activation,
                nn.Linear(projection_channels, out_channels),
            )
        else:
            self.proj = nn.Linear(hidden_channels, out_channels)

        # Dropout
        if dropout > 0.0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None

        # Store graph connectivity (computed in forward pass or pre-computed)
        self.edge_index = None
        self.edge_weights = None

    def _build_graph(self, pos: torch.Tensor) -> tuple:
        """
        Build graph connectivity based on positions.

        Args:
            pos (torch.Tensor): Node positions [batch, n_points, pos_dim]
                               or [n_points, pos_dim]

        Returns:
            edge_index (torch.Tensor): Edge indices [2, n_edges]
            edge_weights (torch.Tensor): Edge weights [n_edges]
        """
        if self.neighbor_strategy == "radius":
            return radius_graph(
                pos, r=self.radius, max_num_neighbors=self.k_neighbors or 32
            )
        elif self.neighbor_strategy == "knn":
            edge_index = knn_graph(pos, k=self.k_neighbors or 8)
            return edge_index, None
        else:
            raise ValueError(f"Unknown neighbor strategy: {self.neighbor_strategy}")

    def forward(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the GNO model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, in_channels, n_points]
                             or [batch, n_points, in_channels]
            pos (torch.Tensor, optional): Node positions [batch, n_points, pos_dim]
                                         or [n_points, pos_dim]. Required if add_grid=True.
            edge_index (torch.Tensor, optional): Pre-computed edge indices [2, n_edges].
                                               If None, computed from positions.
            edge_weights (torch.Tensor, optional): Edge weights [n_edges].

        Returns:
            torch.Tensor: Output tensor of shape [batch, out_channels, n_points]
                         or [batch, n_points, out_channels]
        """
        # Determine input format
        if x.dim() == 2:
            raise ValueError("Input tensor must have batch dimension")

        # Handle different input formats: [batch, channels, n_points] -> [batch, n_points, channels]
        if x.shape[1] == self.in_channels:
            x = x.permute(0, 2, 1)  # [batch, n_points, channels]

        batch_size, n_points, _ = x.shape

        # Handle positions
        if pos is None:
            # Generate default grid positions if not provided
            pos = self._default_positions(n_points, x.device, x.dtype)
        elif pos.dim() == 2:  # [n_points, pos_dim]
            pos = pos.unsqueeze(0).expand(batch_size, -1, -1)
        elif pos.shape[0] == 1 and batch_size > 1:
            pos = pos.expand(batch_size, -1, -1)

        # Add grid coordinates if requested
        if self.add_grid:
            x = torch.cat([x, pos], dim=-1)  # [batch, n_points, in_channels + pos_dim]

        # Lifting layer
        x = self.lift(x)

        # Build graph if not provided
        if edge_index is None:
            edge_index, edge_weights = self._build_graph(pos)

        # Ensure edge_index is on the same device
        edge_index = edge_index.to(x.device)
        if edge_weights is not None:
            edge_weights = edge_weights.to(x.device)

        # GNO layers
        for gno_layer in self.gno_layers:
            x = gno_layer(x, pos, edge_index, edge_weights)

            if self.dropout_layer is not None:
                x = self.dropout_layer(x)

        # Projection layer
        x = self.proj(x)

        # Output format: [batch, n_points, out_channels] -> [batch, out_channels, n_points]
        x = x.permute(0, 2, 1)

        return x

    def _default_positions(
        self, n_points: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Generate default positions on a regular grid."""
        # Simple 1D line of points
        positions = torch.linspace(0, 1, n_points, device=device, dtype=dtype)
        positions = positions.unsqueeze(-1)  # [n_points, 1]
        return positions


class GNO2D(GNO):
    """
    Graph Neural Operator specialized for 2D irregular meshes.
    """

    def _default_positions(
        self, n_points: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Generate default 2D grid positions."""
        # Square root for roughly square grid
        grid_size = int(
            torch.sqrt(torch.tensor(n_points, device=device)).floor().item()
        )

        # Create 2D grid
        x = torch.linspace(0, 1, grid_size, device=device, dtype=dtype)
        y = torch.linspace(0, 1, grid_size, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(x, y, indexing="ij")

        # Flatten and take first n_points
        positions = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        if positions.shape[0] > n_points:
            positions = positions[:n_points]
        elif positions.shape[0] < n_points:
            # Pad with last position if needed
            padding = torch.full(
                (n_points - positions.shape[0], 2),
                positions[-1].item(),
                device=device,
                dtype=dtype,
            )
            positions = torch.cat([positions, padding], dim=0)

        return positions


class GNO3D(GNO):
    """
    Graph Neural Operator specialized for 3D irregular meshes.
    """

    def _default_positions(
        self, n_points: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Generate default 3D grid positions."""
        # Cube root for roughly cubic grid
        grid_size = int(torch.tensor(n_points, device=device).pow(1 / 3).floor().item())

        # Create 3D grid
        x = torch.linspace(0, 1, grid_size, device=device, dtype=dtype)
        y = torch.linspace(0, 1, grid_size, device=device, dtype=dtype)
        z = torch.linspace(0, 1, grid_size, device=device, dtype=dtype)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")

        # Flatten and take first n_points
        positions = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)
        if positions.shape[0] > n_points:
            positions = positions[:n_points]
        elif positions.shape[0] < n_points:
            padding = torch.full(
                (n_points - positions.shape[0], 3),
                positions[-1].item(),
                device=device,
                dtype=dtype,
            )
            positions = torch.cat([positions, padding], dim=0)

        return positions


if __name__ == "__main__":
    # Test the GNO model
    model = GNO(
        in_channels=3,
        out_channels=1,
        num_gno_layers=2,
        hidden_channels=32,
        lifting_channels=32,
        projection_channels=32,
        activation=nn.GELU(),
        radius=0.2,
        k_neighbors=16,
        neighbor_strategy="knn",
        add_grid=True,
        skip="skip",
        dropout=0.1,
    )

    # Test input
    batch_size = 4
    n_points = 64
    x = torch.randn(batch_size, 3, n_points)
    pos = torch.rand(batch_size, n_points, 2)

    output = model(x, pos=pos)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: torch.Size([{batch_size}, 1, {n_points}])")
