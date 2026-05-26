import torch
import torch.nn as nn

from nops.fno.models.original import FNO


def _format_size(num_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024


class FNOConfig:
    def __init__(self, modes=None, num_fourier_layers=2, in_channels=2,
                 lifting_channels=16, projection_channels=16, out_channels=1,
                 mid_channels=16, activation=None, add_grid=True,
                 n_fno_blocks_per_layer=1, dropout=0.05):
        self.modes = modes or [8, 8, 8]
        self.num_fourier_layers = num_fourier_layers
        self.in_channels = in_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.activation = activation or nn.GELU()
        self.add_grid = add_grid
        self.n_fno_blocks_per_layer = n_fno_blocks_per_layer
        self.dropout = dropout

    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        activation = params.pop("activation", None)
        if activation in (None, "gelu", "GELU", "nn.GELU"):
            activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        return cls(activation=activation, **params)


def create_model(config: FNOConfig, device: torch.device) -> nn.Module:
    model = FNO(
        modes=config.modes,
        num_fourier_layers=config.num_fourier_layers,
        in_channels=config.in_channels,
        lifting_channels=config.lifting_channels,
        projection_channels=config.projection_channels,
        out_channels=config.out_channels,
        mid_channels=config.mid_channels,
        activation=config.activation,
        add_grid=config.add_grid,
        n_fno_blocks_per_layer=config.n_fno_blocks_per_layer,
        dropout=config.dropout
    )
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    print(f"  FNO3D parameters: {params:,} ({_format_size(model_bytes)})")
    return model
