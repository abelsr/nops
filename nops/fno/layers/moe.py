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
            
            # Renormalize weights to sum to 1
            top_k_weights = F.softmax(top_k_weights, dim=-1)
            
            # Run only top-k experts
            # Create output tensor with same shape as input
            output = torch.zeros_like(x)
            
            # For each batch sample, run only the selected experts
            for b in range(batch_size):
                for k_idx in range(self.top_k):
                    expert_idx = top_k_indices[b, k_idx].item()
                    expert_weight = top_k_weights[b, k_idx]
                    
                    # Run the expert on this sample
                    expert_output = self.experts[expert_idx](x[b:b+1])
                    
                    # Add weighted contribution
                    output[b:b+1] += expert_weight * expert_output
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
