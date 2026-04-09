"""
Utilities for working with irregular meshes and point clouds.
"""

import torch
from typing import Tuple, Optional


def create_meshgrid_2d(
    n_points: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Create a 2D mesh grid of points.

    Args:
        n_points (int): Target number of points
        device (torch.device): Device for the tensor
        dtype (torch.dtype): Data type for the tensor

    Returns:
        positions (torch.Tensor): Grid positions of shape [n_points, 2]
    """
    grid_size = int(torch.sqrt(torch.tensor(n_points)).floor().item())
    if grid_size * grid_size < n_points:
        grid_size += 1

    x = torch.linspace(0, 1, grid_size, device=device, dtype=dtype)
    y = torch.linspace(0, 1, grid_size, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    positions = torch.stack([xx.flatten(), yy.flatten()], dim=-1)

    if positions.shape[0] > n_points:
        positions = positions[:n_points]
    elif positions.shape[0] < n_points:
        padding = torch.full(
            (n_points - positions.shape[0], 2),
            positions[-1].item(),
            device=device,
            dtype=dtype,
        )
        positions = torch.cat([positions, padding], dim=0)

    return positions


def create_meshgrid_3d(
    n_points: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Create a 3D mesh grid of points.

    Args:
        n_points (int): Target number of points
        device (torch.device): Device for the tensor
        dtype (torch.dtype): Data type for the tensor

    Returns:
        positions (torch.Tensor): Grid positions of shape [n_points, 3]
    """
    grid_size = int(torch.tensor(n_points).pow(1 / 3).floor().item())
    if grid_size**3 < n_points:
        grid_size += 1

    x = torch.linspace(0, 1, grid_size, device=device, dtype=dtype)
    y = torch.linspace(0, 1, grid_size, device=device, dtype=dtype)
    z = torch.linspace(0, 1, grid_size, device=device, dtype=dtype)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")

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


def normalize_positions(
    pos: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0
) -> torch.Tensor:
    """
    Normalize positions to [min_val, max_val] range.

    Args:
        pos (torch.Tensor): Positions of shape [..., dim]
        min_val (float): Minimum value for normalization
        max_val (float): Maximum value for normalization

    Returns:
        normalized (torch.Tensor): Normalized positions
    """
    pos_min = pos.amin(dim=0, keepdim=True)
    pos_max = pos.amax(dim=0, keepdim=True)

    normalized = (pos - pos_min) / (pos_max - pos_min + 1e-8)
    normalized = normalized * (max_val - min_val) + min_val

    return normalized


def compute_edge_features(
    pos: torch.Tensor, edge_index: torch.Tensor, pos_dim: int = 2
) -> torch.Tensor:
    """
    Compute features for edges based on node positions.

    Args:
        pos (torch.Tensor): Node positions [n_points, pos_dim]
        edge_index (torch.Tensor): Edge indices [2, n_edges]
        pos_dim (int): Position dimension

    Returns:
        edge_features (torch.Tensor): Edge features [n_edges, feature_dim]
    """
    src_idx = edge_index[0]
    dst_idx = edge_index[1]

    pos_src = pos[src_idx]
    pos_dst = pos[dst_idx]

    # Relative position
    rel_pos = pos_dst - pos_src

    # Distance
    dist = torch.norm(rel_pos, dim=-1, keepdim=True)

    # Normalized relative position
    rel_pos_normalized = rel_pos / (dist + 1e-8)

    # Combine features
    edge_features = torch.cat([rel_pos, rel_pos_normalized, dist], dim=-1)

    return edge_features


def laplacian_matrix(
    edge_index: torch.Tensor, n_nodes: int, edge_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute the graph Laplacian matrix.

    Args:
        edge_index (torch.Tensor): Edge indices [2, n_edges]
        n_nodes (int): Number of nodes
        edge_weights (torch.Tensor, optional): Edge weights [n_edges]

    Returns:
        laplacian (torch.Tensor): Graph Laplacian [n_nodes, n_nodes]
    """
    laplacian = torch.zeros(n_nodes, n_nodes)

    src_idx = edge_index[0]
    dst_idx = edge_index[1]

    if edge_weights is None:
        edge_weights = torch.ones(edge_index.shape[1])

    for i, (src, dst) in enumerate(zip(src_idx, dst_idx)):
        laplacian[src, dst] -= edge_weights[i]
        laplacian[dst, src] -= edge_weights[i]

    # Add diagonal (degree matrix)
    degree = laplacian.abs().sum(dim=1)
    laplacian += torch.diag(degree)

    return laplacian
