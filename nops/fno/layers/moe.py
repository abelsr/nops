import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class Router(nn.Module):
    """
    Router for Mixture of Experts.
    Supports global routing (per sample) or patch routing (per spatial location).
    """
    def __init__(self, in_channels: int, num_experts: int, temperature: float = 1.0, k: int = 1, routing_type: str = 'global'):
        super().__init__()
        self.in_channels = in_channels
        self.num_experts = num_experts
        self.temperature = temperature
        self.k = min(k, num_experts)
        self.routing_type = routing_type
        self.gate = nn.Linear(in_channels, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, *spatial_dims]
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Expert weights: [batch, k] if global, [batch, *spatial_dims, k] if patch
                - Expert indices: [batch, k] if global, [batch, *spatial_dims, k] if patch
        """
        if self.routing_type == 'global':
            # Global Average Pooling
            if x.dim() > 2:
                reduce_dims = tuple(range(2, x.dim()))
                pooled = x.mean(dim=reduce_dims)  # [batch, channels]
            else:
                pooled = x
            logits = self.gate(pooled)  # [batch, num_experts]
        elif self.routing_type == 'patch':
            # Permute to [batch, *spatial_dims, channels]
            # x is [batch, channels, d1, d2, ...]
            dims = list(range(x.dim()))
            # Move channels (dim 1) to the end
            permute_order = [dims[0]] + dims[2:] + [dims[1]]
            x_permuted = x.permute(*permute_order)
            logits = self.gate(x_permuted)  # [batch, *spatial_dims, num_experts]
        else:
            raise ValueError(f"Unknown routing_type: {self.routing_type}")
        
        if self.temperature != 1.0:
            logits = logits / self.temperature
            
        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=-1)
        weights = F.softmax(top_k_logits, dim=-1)
        
        return weights, top_k_indices

class MoEBlock(nn.Module):
    """
    Mixture of Experts Block with Top-K and Patch/Global routing support.
    """
    def __init__(self, experts: nn.ModuleList, router: Router, top_k: Optional[int] = None):
        super().__init__()
        self.experts = experts
        self.router = router
        self.num_experts = len(experts)
        self.top_k = top_k if top_k is not None else self.num_experts
        
        # Validate top_k
        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError(f"top_k must be between 1 and {self.num_experts}, got {self.top_k}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, *spatial_dims]
        Returns:
            torch.Tensor: Output tensor of shape [batch, channels, *spatial_dims]
        """
        weights, indices = self.router(x)
        # weights, indices: [batch, k] or [batch, *spatial_dims, k]
        
        output = torch.zeros_like(x)
        
        if self.router.routing_type == 'global':
            # Optimization: Only run experts that are selected for at least one sample in the batch
            for i, expert in enumerate(self.experts):
                mask = (indices == i) # [batch, k]
                if mask.any():
                    batch_idx, k_idx = torch.where(mask)
                    
                    # Run expert on selected samples
                    expert_input = x[batch_idx]
                    expert_output = expert(expert_input)
                    
                    # Apply weights
                    w = weights[batch_idx, k_idx]
                    for _ in range(expert_output.dim() - 1):
                        w = w.unsqueeze(-1)
                    
                    output.index_add_(0, batch_idx, expert_output * w)
        
        elif self.router.routing_type == 'patch':
            # For patch routing, we run experts that are selected at least once in the batch
            # and apply weights spatially.
            for i, expert in enumerate(self.experts):
                mask = (indices == i) # [batch, *spatial_dims, k]
                if mask.any():
                    # Run expert on the full input (required for FNO)
                    expert_output = expert(x) # [batch, channels, *spatial_dims]
                    
                    # weights[mask] gives the weights for expert i where it was selected
                    # We need to scatter these weights back to a spatial mask
                    
                    # Create a spatial weight map for this expert
                    # spatial_weights: [batch, *spatial_dims]
                    spatial_weights = torch.zeros(indices.shape[:-1], device=x.device, dtype=x.dtype)
                    
                    # Sum weights for this expert across the k dimension (in case it's selected multiple times, 
                    # though top-k usually ensures unique indices)
                    spatial_weights.masked_scatter_(mask.any(dim=-1), weights[mask])
                    
                    # Multiply expert output by spatial weights
                    # spatial_weights: [batch, *spatial_dims] -> [batch, 1, *spatial_dims]
                    output += expert_output * spatial_weights.unsqueeze(1)
                    
        return output
