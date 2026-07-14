from typing import List, Literal

import torch.nn as nn

from nops.fno.models.original import FNO


class NavierStokesFNO(FNO):
    """FNO model configured for 2D incompressible Navier-Stokes."""

    def __init__(
        self,
        dimension: Literal["2D", "3D"],
        modes: List[int],
        num_fourier_layers: int,
        in_channels: int,
        lifting_channels: int,
        projection_channels: int,
        out_channels: int,
        mid_channels: int,
        activation: nn.Module,
        **kwargs,
    ) -> None:
        self.dimension = dimension
        if self.dimension == "2D":
            assert len(modes) == 2
        elif self.dimension == "3D":
            assert len(modes) == 3
        else:
            raise ValueError("dimension must be '2D' or '3D'.")

        super().__init__(
            modes=modes,
            num_fourier_layers=num_fourier_layers,
            in_channels=in_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            out_channels=out_channels,
            mid_channels=mid_channels,
            activation=activation,
            add_grid=kwargs.get("add_grid", True),
            spectral_norm=kwargs.get("spectral_norm", False),
            residual=kwargs.get("residual", False),
            resolution_aware=kwargs.get("resolution_aware", True),
            num_resolutions=kwargs.get("num_resolutions", 3),
            dropout=kwargs.get("dropout", 0.0),
            attn_gating=kwargs.get("attn_gating", False),
            n_fno_blocks_per_layer=kwargs.get("n_fno_blocks_per_layer", 1),
            padding=kwargs.get("padding", None),
        )

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
