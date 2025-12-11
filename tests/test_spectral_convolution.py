import torch
import pytest

from nops.fno.layers.spectral_convolution import SpectralConvolution
from nops.fno.layers.fno_block import FourierBlock


def test_spectral_convolution_forward_shape_and_finiteness():
    """Dense spectral conv should run on CPU and preserve spatial dims."""
    torch.manual_seed(0)
    layer = SpectralConvolution(
        in_channels=4,
        out_channels=4,
        modes=[4, 4],
        factorization="dense",
        bias=True,
    )
    x = torch.randn(2, 4, 8, 8)

    out = layer(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_spectral_convolution_raises_on_dim_mismatch():
    layer = SpectralConvolution(
        in_channels=2,
        out_channels=2,
        modes=[4, 4],
        factorization="dense",
    )
    bad_input = torch.randn(1, 2, 8)  # Missing one spatial dimension

    with pytest.raises(ValueError):
        _ = layer(bad_input)


def test_fourier_block_backward_pass():
    """Ensure gradients flow through FourierBlock components."""
    torch.manual_seed(0)
    block = FourierBlock(
        modes=[4, 4],
        in_channels=3,
        out_channels=5,
        hidden_size=6,
        activation=torch.nn.ReLU(),
    )
    x = torch.randn(2, 3, 8, 8, requires_grad=True)

    out = block(x)
    assert out.shape == (2, 5, 8, 8)

    out.mean().backward()
    assert x.grad is not None


def test_spectral_convolution_bias_broadcast_without_residual():
    """Bias should broadcast even when in/out channels differ."""
    layer = SpectralConvolution(
        in_channels=2,
        out_channels=3,
        modes=[2, 2],
        factorization="dense",
        bias=True,
    )
    with torch.no_grad():
        layer.weights_real.zero_()
        layer.weights_imag.zero_()
        layer.bias.fill_(0.25)

    x = torch.zeros(1, 2, 4, 4)
    out = layer(x)

    expected = torch.full((1, 3, 4, 4), 0.25)
    assert torch.allclose(out, expected)
