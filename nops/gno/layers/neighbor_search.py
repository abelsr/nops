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

    # Compute pairwise distances
    # Using efficient matrix multiplication for squared distances
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * <x_i, x_j>
    pos_squared = torch.sum(pos_flat**2, dim=1, keepdim=True)  # [N, 1]
    pos_dot = torch.matmul(pos_flat, pos_flat.t())  # [N, N]
    dist_squared = pos_squared + pos_squared.t() - 2 * pos_dot  # [N, N]

    # Clamp to avoid negative values due to numerical instability
    dist_squared = torch.clamp(dist_squared, min=0.0)
    dist = torch.sqrt(dist_squared + 1e-12)  # [N, N]

    # Build edges per batch item
    edge_src_list = []
    edge_dst_list = []
    edge_dist_list = []

    for b in range(batch_size):
        # Get valid neighbors for each node in this batch
        for i in range(n_points):
            for j in range(n_points):
                if i == j and not loop:
                    continue
                d = dist[b * n_points + i, b * n_points + j].item()
                if d <= r:
                    edge_src_list.append(b * n_points + i)
                    edge_dst_list.append(b * n_points + j)
                    edge_dist_list.append(d)

    if not edge_src_list:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=pos.device)
        edge_weights = torch.empty((0,), dtype=pos.dtype, device=pos.device)
        return edge_index, edge_weights

    # Convert to tensors
    edge_index = torch.tensor(
        [edge_src_list, edge_dst_list], device=pos.device, dtype=torch.long
    )
    edge_weights = torch.tensor(edge_dist_list, device=pos.device, dtype=pos.dtype)

    # Limit to max_num_neighbors per node
    if max_num_neighbors > 0:
        # For each source node, keep only the closest max_num_neighbors
        unique_src, inverse_idx = torch.unique(edge_index[0], return_inverse=True)

        filtered_src = []
        filtered_dst = []
        filtered_weights = []

        for i, src in enumerate(unique_src):
            mask = edge_index[0] == src
            src_edges = edge_index[1][mask]
            src_weights = edge_weights[mask]

            if src_edges.shape[0] > max_num_neighbors:
                # Keep k smallest
                _, topk_idx = torch.topk(src_weights, max_num_neighbors, largest=False)
                filtered_src.extend([src.item()] * max_num_neighbors)
                filtered_dst.extend(src_edges[topk_idx].tolist())
                filtered_weights.extend(src_weights[topk_idx].tolist())
            else:
                filtered_src.extend([src.item()] * src_edges.shape[0])
                filtered_dst.extend(src_edges.tolist())
                filtered_weights.extend(src_weights.tolist())

        edge_index = torch.tensor(
            [filtered_src, filtered_dst], device=pos.device, dtype=torch.long
        )
        edge_weights = torch.tensor(
            filtered_weights, device=pos.device, dtype=pos.dtype
        )

    return edge_index, edge_weights

    # Convert to tensors
    edge_index = torch.tensor(valid_connections, device=pos.device).t()  # [2, n_edges]
    edge_weights = torch.tensor(
        [c[2] for c in valid_connections], device=pos.device, dtype=pos.dtype
    )  # [n_edges]

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
