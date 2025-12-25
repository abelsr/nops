import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class Router(nn.Module):
    """
    Router for Mixture of Experts.
    It uses Global Average Pooling to get a global representation of the input
    and then a linear layer to compute the weights for each expert.
    """
    def __init__(self, in_channels: int, num_experts: int, temperature: float = 1.0):
        super().__init__()
        self.in_channels = in_channels
        self.num_experts = num_experts
        self.temperature = temperature
        self.gate = nn.Linear(in_channels, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, *spatial_dims]
        Returns:
            torch.Tensor: Expert weights of shape [batch, num_experts]
        """
        # Global Average Pooling
        if x.dim() > 2:
            reduce_dims = tuple(range(2, x.dim()))
            pooled = x.mean(dim=reduce_dims)  # [batch, channels]
        else:
            pooled = x

        logits = self.gate(pooled)  # [batch, num_experts]
        
        if self.temperature != 1.0:
            logits = logits / self.temperature
            
        weights = F.softmax(logits, dim=-1)
        return weights

class MoEBlock(nn.Module):
    """
    Mixture of Experts Block.
    
    Args:
        experts: ModuleList of expert networks
        router: Router module to compute expert weights
        top_k: Number of top experts to use. If None, uses all experts. Default: None
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
        weights = self.router(x)  # [batch, num_experts]
        batch_size = weights.shape[0]
        
        # Top-K expert selection
        if self.top_k < self.num_experts:
            # Select top-k experts per sample
            top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)  # [batch, top_k]
            
            # Renormalize weights to sum to 1 (with epsilon for numerical stability)
            top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-10)
            
            # Efficient batch processing: group samples by expert
            # Create a sparse representation to avoid running all experts
            output = torch.zeros_like(x)
            
            # Process each expert once with all samples that need it
            for expert_idx in range(self.num_experts):
                # Find which samples use this expert
                mask = (top_k_indices == expert_idx).any(dim=-1)  # [batch]
                if not mask.any():
                    continue  # Skip if no samples use this expert
                
                # Get the samples that use this expert
                sample_indices = mask.nonzero(as_tuple=True)[0]  # 1D tensor of sample indices
                
                # Run expert on selected samples
                x_subset = x[sample_indices]
                expert_output = self.experts[expert_idx](x_subset)
                
                # Find weights for this expert in each sample
                for i, sample_idx in enumerate(sample_indices):
                    # Find position of this expert in the top-k for this sample
                    k_positions = (top_k_indices[sample_idx] == expert_idx).nonzero(as_tuple=True)[0]
                    
                    for k_pos in k_positions:
                        weight = top_k_weights[sample_idx, k_pos]
                        output[sample_idx] += weight * expert_output[i]
        else:
            # Run all experts (original behavior)
            expert_outputs = [expert(x) for expert in self.experts]
            expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, channels, *spatial_dims]
            
            # Weighted sum
            # weights: [batch, num_experts] -> [batch, num_experts, 1, 1, ...]
            expand_shape = [weights.shape[0], weights.shape[1]] + [1] * (expert_outputs.dim() - 2)
            weights = weights.view(*expand_shape)
            
            output = (expert_outputs * weights).sum(dim=1)
        
        return output
