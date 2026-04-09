"""Test DeepONet implementation."""

import torch
import torch.nn as nn
from nops.deeponet.models import DeepONet, DeepONetCartesianProd


def test_deeponet_basic():
    """Test basic DeepONet forward pass."""
    print("Testing basic DeepONet...")

    model = DeepONet(
        branch_layers=[10, 32, 32],
        trunk_layers=[1, 32, 32],  # 1D trunk input
        activation=nn.GELU(),
        num_outputs=1,
    )

    # Test with 2D inputs (batch_size, sensors/points)
    batch_size = 4
    branch_input = torch.randn(batch_size, 10)  # 10 sensor values per sample
    trunk_input = torch.randn(batch_size, 5)  # 5 query points (1D coordinates)

    output = model(branch_input, trunk_input)

    assert output.shape == (batch_size, 5, 1), (
        f"Expected {(batch_size, 5, 1)}, got {output.shape}"
    )
    print(f"  Output shape: {output.shape} ✓")


def test_deeponet_3d():
    """Test DeepONet with 3D inputs."""
    print("Testing DeepONet with 3D inputs...")

    model = DeepONet(
        branch_layers=[10, 32, 32],
        trunk_layers=[2, 32, 32],
        activation=nn.GELU(),
        num_outputs=1,
    )

    batch_size = 4
    branch_input = torch.randn(batch_size, 10, 1)  # 10 sensors, 1 feature per sensor
    trunk_input = torch.randn(batch_size, 5, 2)  # 5 query points in 2D

    output = model(branch_input, trunk_input)

    assert output.shape == (batch_size, 5, 1), (
        f"Expected {(batch_size, 5, 1)}, got {output.shape}"
    )
    print(f"  Output shape: {output.shape} ✓")


def test_deeponet_multiple_outputs():
    """Test DeepONet with multiple outputs."""
    print("Testing DeepONet with multiple outputs...")

    model = DeepONet(
        branch_layers=[10, 32, 32],
        trunk_layers=[1, 32, 32],  # 1D trunk input
        activation=nn.GELU(),
        num_outputs=3,
    )

    batch_size = 4
    branch_input = torch.randn(batch_size, 10)
    trunk_input = torch.randn(batch_size, 5)

    output = model(branch_input, trunk_input)

    assert output.shape == (batch_size, 5, 3), (
        f"Expected {(batch_size, 5, 3)}, got {output.shape}"
    )
    print(f"  Output shape: {output.shape} ✓")


def test_deeponet_cartesian():
    """Test DeepONetCartesianProd."""
    print("Testing DeepONetCartesianProd...")

    model = DeepONetCartesianProd(
        branch_layers=[10, 32, 32],
        trunk_layers=[2, 32, 32],
        activation=nn.GELU(),
        num_outputs=1,
    )

    batch_size = 4
    branch_input = torch.randn(batch_size, 10)  # [batch, branch_dim]
    trunk_input = torch.randn(batch_size * 5, 2)  # [batch * num_points, trunk_dim]

    output = model(branch_input, trunk_input)

    assert output.shape == (batch_size, 5, 1), (
        f"Expected {(batch_size, 5, 1)}, got {output.shape}"
    )
    print(f"  Output shape: {output.shape} ✓")


def test_deeponet_encode_methods():
    """Test encode_branch and encode_trunk methods."""
    print("Testing encode methods...")

    model = DeepONet(
        branch_layers=[10, 32, 32], trunk_layers=[2, 32, 32], activation=nn.GELU()
    )

    batch_size = 4
    branch_input = torch.randn(batch_size, 10, 1)
    trunk_input = torch.randn(batch_size, 5, 2)

    branch_encoded = model.encode_branch(branch_input)
    trunk_encoded = model.encode_trunk(trunk_input)

    assert branch_encoded.shape == (batch_size, 32), (
        f"Branch encoded: expected {(batch_size, 32)}, got {branch_encoded.shape}"
    )
    assert trunk_encoded.shape == (batch_size, 5, 32), (
        f"Trunk encoded: expected {(batch_size, 5, 32)}, got {trunk_encoded.shape}"
    )
    print(f"  Branch encoded shape: {branch_encoded.shape} ✓")
    print(f"  Trunk encoded shape: {trunk_encoded.shape} ✓")


if __name__ == "__main__":
    print("Running DeepONet tests...\n")

    test_deeponet_basic()
    test_deeponet_3d()
    test_deeponet_multiple_outputs()
    test_deeponet_cartesian()
    test_deeponet_encode_methods()

    print("\n✓ All tests passed!")
