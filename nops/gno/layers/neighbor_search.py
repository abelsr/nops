"""
Neighborhood search utilities for Graph Neural Operators.
"""

import torch
from typing import Tuple, Optional


def radius_graph(
    pos: torch.Tensor,
    r: float,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute graph connectivity based on radial distance.

    Args:
        pos (torch.Tensor): Node positions of shape [n_points, pos_dim]
                           or [batch_size, n_points, pos_dim]
        r (float): Radius for neighbor search
        batch (torch.Tensor, optional): Batch assignment for each node
                                      of shape [n_points] or [batch_size * n_points]
                                      If None, all nodes belong to the same graph
        loop (bool): If True, include self-loops in the graph
        max_num_neighbors (int): Maximum number of neighbors to return per node

    Returns:
        edge_index (torch.Tensor): Edge indices of shape [2, n_edges]
        edge_weights (torch.Tensor): Edge weights (distances) of shape [n_edges]
    """
    # Handle input dimensions
    if pos.dim() == 2:  # [n_points, pos_dim]
        n_points = pos.shape[0]
        pos_dim = pos.shape[1]
        batch_size = 1
        # Expand to batch format for uniform handling
        pos = pos.unsqueeze(0)  # [1, n_points, pos_dim]
        if batch is None:
            batch = torch.zeros(n_points, dtype=torch.long, device=pos.device)
        else:
            batch = batch.unsqueeze(0)  # [1, n_points]
    elif pos.dim() == 3:  # [batch_size, n_points, pos_dim]
        batch_size, n_points, pos_dim = pos.shape
        if batch is None:
            # Create batch tensor: [batch_size * n_points]
            batch = torch.arange(batch_size, device=pos.device)
            batch = batch.unsqueeze(-1).expand(-1, n_points).reshape(-1)
        else:
            if batch.dim() == 1:
                # Assume batch is [n_points] and expand to batch dimension
                batch = batch.unsqueeze(0).expand(batch_size, -1).reshape(-1)
            elif batch.dim() == 2 and batch.shape[0] == batch_size:
                # Batch is [batch_size, n_points]
                batch = batch.reshape(-1)
    else:
        raise ValueError(f"Position tensor must be 2D or 3D, got {pos.dim()}D")

    # Flatten positions for easier computation
    # pos_flat: [batch_size * n_points, pos_dim]
    pos_flat = pos.view(-1, pos_dim)
    batch_flat = batch  # [batch_size * n_points]

    # Compute pairwise distance matrix using cdist
    N = pos_flat.size(0)
    dist = torch.cdist(pos_flat, pos_flat)  # [N, N]

    # Mask distances: only keep same-batch pairs, set others to inf
    dist_masked = torch.full((N, N), float("inf"), device=pos.device, dtype=pos.dtype)
    batch_eq = batch_flat.unsqueeze(0) == batch_flat.unsqueeze(1)  # [N, N]
    dist_masked = torch.where(batch_eq, dist, dist_masked)

    # Exclude self-loops if requested
    if not loop:
        self_mask = torch.eye(N, dtype=torch.bool, device=pos.device)
        dist_masked = dist_masked.masked_fill(self_mask, float("inf"))

    # Build edges
    if max_num_neighbors > 0:
        # Use topk per node to get at most max_num_neighbors closest neighbors.
        # When self-loops are allowed, a node may select itself, so up to N
        # neighbors are possible. Otherwise the maximum is N - 1.
        max_possible_neighbors = N if loop else max(N - 1, 1)
        k = min(max_num_neighbors, max_possible_neighbors)
        weights, indices = torch.topk(dist_masked, k, dim=1, largest=False)

        # Keep only neighbors within radius (filter out inf entries)
        valid = weights <= r
        src = torch.arange(N, device=pos.device).unsqueeze(1).expand(-1, k)
        src = src[valid]
        dst = indices[valid]
        edge_weights = weights[valid]
    else:
        # No limit: find all pairs within radius
        src, dst = torch.where(dist_masked <= r)
        edge_weights = dist_masked[src, dst]

    if src.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=pos.device)
        edge_weights = torch.empty((0,), dtype=pos.dtype, device=pos.device)
        return edge_index, edge_weights

    edge_index = torch.stack([src, dst], dim=0)  # [2, n_edges]

    return edge_index, edge_weights


def knn_graph(
    pos: torch.Tensor,
    k: int,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    flow: str = "source_to_target",
) -> torch.Tensor:
    """
    Compute k-nearest neighbor graph connectivity.

    Args:
        pos (torch.Tensor): Node positions of shape [n_points, pos_dim]
                           or [batch_size, n_points, pos_dim]
        k (int): Number of nearest neighbors
        batch (torch.Tensor, optional): Batch assignment for each node
        loop (bool): If True, include self-loops
        flow (str): Flow direction ('source_to_target' or 'target_to_source')
                   Determines the order of indices in edge_index

    Returns:
        edge_index (torch.Tensor): Edge indices of shape [2, n_edges]
    """
    # Handle input dimensions
    if pos.dim() == 2:  # [n_points, pos_dim]
        n_points = pos.shape[0]
        pos_dim = pos.shape[1]
        batch_size = 1
        pos = pos.unsqueeze(0)  # [1, n_points, pos_dim]
        if batch is None:
            batch = torch.zeros(n_points, dtype=torch.long, device=pos.device)
        else:
            batch = batch.unsqueeze(0)  # [1, n_points]
    elif pos.dim() == 3:  # [batch_size, n_points, pos_dim]
        batch_size, n_points, pos_dim = pos.shape
        if batch is None:
            batch = torch.arange(batch_size, device=pos.device)
            batch = batch.unsqueeze(-1).expand(-1, n_points).reshape(-1)
        else:
            if batch.dim() == 1:
                batch = batch.unsqueeze(0).expand(batch_size, -1).reshape(-1)
            elif batch.dim() == 2 and batch.shape[0] == batch_size:
                batch = batch.reshape(-1)
    else:
        raise ValueError(f"Position tensor must be 2D or 3D, got {pos.dim()}D")

    # For simplicity, compute KNN per-batch-item separately
    # This returns edge indices that are valid within each batch item
    edge_indices = []

    for b in range(batch_size):
        pos_b = pos[b]  # [n_points, pos_dim]

        # Compute pairwise distances for this batch item
        pos_squared = torch.sum(pos_b**2, dim=1, keepdim=True)  # [n_points, 1]
        pos_dot = torch.matmul(pos_b, pos_b.t())  # [n_points, n_points]
        dist_squared = pos_squared + pos_squared.t() - 2 * pos_dot
        dist_squared = torch.clamp(dist_squared, min=0.0)
        dist = torch.sqrt(dist_squared + 1e-12)  # [n_points, n_points]

        # Set self-distance to inf if loops not allowed
        if not loop:
            dist.fill_diagonal_(float("inf"))

        # Get k nearest neighbors for each node
        _, knn_indices = torch.topk(dist, k=min(k, n_points), dim=1, largest=False)

        # Create edges for this batch
        if flow == "source_to_target":
            src = torch.arange(n_points, device=pos.device).unsqueeze(1).expand(-1, k)
            dst = knn_indices
        else:
            src = knn_indices
            dst = torch.arange(n_points, device=pos.device).unsqueeze(1).expand(-1, k)

        # Flatten and append to list
        edge_indices.append(torch.stack([src.flatten(), dst.flatten()], dim=0))

    # Concatenate all edges
    edge_index = torch.cat(edge_indices, dim=1)  # [2, total_edges]

    return edge_index
