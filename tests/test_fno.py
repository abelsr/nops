import torch
import torch.nn as nn

from nops.fno.models.original import FNO


def test_fno_forward_shape_with_grid_padding_and_attention():
    torch.manual_seed(0)
    model = FNO(
        modes=[4, 4],
        num_fourier_layers=2,
        in_channels=3,
        lifting_channels=8,
        projection_channels=6,
        out_channels=2,
        mid_channels=8,
        activation=nn.ReLU(),
        add_grid=True,
        padding=[2, 2],
        n_fno_blocks_per_layer=2,
        dropout=0.0,
        attn_gating=True,
        attn_temperature=1.0,
    )
    x = torch.randn(2, 3, 12, 12)

    out = model(x)

    assert out.shape == (2, 2, 12, 12)

    out.mean().backward()
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()
