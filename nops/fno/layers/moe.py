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
    """
    def __init__(self, experts: nn.ModuleList, router: Router):
        super().__init__()
        self.experts = experts
        self.router = router

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, *spatial_dims]
        Returns:
            torch.Tensor: Output tensor of shape [batch, channels, *spatial_dims]
        """
        weights = self.router(x)  # [batch, num_experts]
        
        # Run all experts
        # TODO: Optimization for Top-K if needed
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, channels, *spatial_dims]
        
        # Weighted sum
        # weights: [batch, num_experts] -> [batch, num_experts, 1, 1, ...]
        expand_shape = [weights.shape[0], weights.shape[1]] + [1] * (expert_outputs.dim() - 2)
        weights = weights.view(*expand_shape)
        
        output = (expert_outputs * weights).sum(dim=1)
        return output
