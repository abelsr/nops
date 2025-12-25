from typing import Any, List 

import torch
import torch.nn as nn
import torch.nn.functional as F

from nops.fno.layers.mlp import MLP
from nops.fno.layers.moe import MoEBlock, Router
from nops.fno.layers.sequential_fno_block import SequentialFourierBlock

class MoEFNO(nn.Module):
    """
    Mixture of Experts Fourier Neural Operator (MoE-FNO).
    """
    def __init__(
        self,
        modes: List[int],
        num_moe_layers: int,
        num_experts: int,
        in_channels: int,
        lifting_channels: int,
        projection_channels: int,
        out_channels: int,
        mid_channels: int,
        expert_hidden_size: int,
        activation: nn.Module = nn.GELU(),
        add_grid: bool = True,
        temperature: float = 1.0,
        **kwargs: Any
    ):
        super().__init__()
        self.modes = modes
        self.dim = len(modes)
        self.num_moe_layers = num_moe_layers
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.activation = activation
        self.add_grid = add_grid
        self.padding = kwargs.get('padding', None)
        
        # Grid setup
        self.sizes = [0] * self.dim
        self.grids = None

        # 1. Lifting Layer (MLP)
        # Input: [batch, *sizes, in_channels + dim]
        # Output: [batch, *sizes, mid_channels]
        lifting_in = in_channels + (self.dim if add_grid else 0)
        self.lifting = nn.Sequential(
            nn.Linear(lifting_in, lifting_channels),
            activation,
            nn.Linear(lifting_channels, mid_channels)
        )

        # 2. MoE Layers
        self.moe_layers = nn.ModuleList([
            MoEBlock(
                experts=nn.ModuleList([
                    SequentialFourierBlock(
                        modes=modes,
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        hidden_size=expert_hidden_size,
                        activation=activation
                    ) for _ in range(num_experts)
                ]),
                router=Router(mid_channels, num_experts, temperature=temperature)
            ) for _ in range(num_moe_layers)
        ])

        # 3. Projection Layer (MLP)
        # Input: [batch, *sizes, mid_channels]
        # Output: [batch, *sizes, out_channels]
        self.projection = nn.Sequential(
            nn.Linear(mid_channels, projection_channels),
            activation,
            nn.Linear(projection_channels, out_channels)
        )

        # Padding setup
        if self.padding is not None:
            # Padding is [pad_dim1, pad_dim2, ...]
            # Convert to torch.pad format: (left, right, top, bottom, front, back)
            self.torch_padding = []
            for p in reversed(self.padding):
                self.torch_padding.extend([p, p])
            
            # Slice for removing padding
            self.slice = tuple(slice(p, -p) if p > 0 else slice(None) for p in self.padding)

    def set_grid(self, x: torch.Tensor) -> None:
        batch, *sizes, _ = x.size()
        self.sizes = sizes
        grids = []
        for i, size in enumerate(sizes):
            grid = torch.linspace(0, 1, size, device=x.device)
            # Reshape to [1, 1, ..., size, ..., 1]
            shape = [1] * (self.dim + 2)
            shape[i + 1] = size
            grid = grid.view(*shape)
            # Repeat to [batch, *sizes, 1]
            repeat_shape = [batch] + sizes + [1]
            repeat_shape[i + 1] = 1
            grid = grid.repeat(*repeat_shape)
            grids.append(grid)
        self.grids = torch.cat(grids, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, in_channels, *sizes]
        batch, _, *sizes = x.shape
        
        # Permute to [batch, *sizes, in_channels]
        x = x.permute(0, *range(2, self.dim + 2), 1)

        # Add grid
        if self.add_grid:
            if self.grids is None or self.grids.shape[0] != batch or list(self.grids.shape[1:-1]) != list(sizes):
                self.set_grid(x)
            x = torch.cat([x, self.grids], dim=-1)

        # Lifting
        x = self.lifting(x)

        # Permute to [batch, mid_channels, *sizes] for FNO blocks
        x = x.permute(0, -1, *range(1, self.dim + 1))

        # Padding
        if self.padding is not None:
            x = F.pad(x, self.torch_padding)

        # MoE Layers
        for moe_layer in self.moe_layers:
            x = moe_layer(x)

        # Unpadding
        if self.padding is not None:
            x = x[(Ellipsis,) + self.slice]

        # Permute back to [batch, *sizes, mid_channels] for projection
        x = x.permute(0, *range(2, self.dim + 2), 1)

        # Projection
        x = self.projection(x)

        # Permute back to [batch, out_channels, *sizes]
        x = x.permute(0, -1, *range(1, self.dim + 1))
        
        return x

if __name__ == "__main__":
    # Test
    model = MoEFNO(
        modes=[8, 8],
        num_moe_layers=2,
        num_experts=4,
        in_channels=1,
        lifting_channels=64,
        projection_channels=64,
        out_channels=1,
        mid_channels=32,
        expert_hidden_size=32,
        padding=[4, 4]
    )
    x = torch.randn(2, 1, 32, 32)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert x.shape == y.shape
