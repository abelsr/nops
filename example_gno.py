"""
Example: Using Graph Neural Operator (GNO) for irregular meshes.

This example demonstrates how to use the GNO model for learning operators
on irregular meshes/point clouds.
"""

import torch
import torch.nn as nn
from nops.gno.models.original import GNO, GNO2D


def main():
    print("=" * 60)
    print("Graph Neural Operator (GNO) Example")
    print("=" * 60)

    # Configuration
    batch_size = 4
    n_points = 64  # Number of points in the mesh/point cloud
    in_channels = 3  # e.g., (u, v, p) - velocity components and pressure
    out_channels = 1  # e.g., pressure field

    # Create GNO model
    model = GNO(
        in_channels=in_channels,
        out_channels=out_channels,
        num_gno_layers=4,
        hidden_channels=64,
        lifting_channels=64,
        projection_channels=64,
        activation=nn.GELU(),
        radius=0.15,  # Radius for neighbor search
        k_neighbors=16,  # Number of neighbors
        neighbor_strategy="knn",  # 'knn' or 'radius'
        add_grid=True,  # Append grid coordinates to input
        skip="skip",
        dropout=0.1,
    )

    print(
        f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Create random input (function values at mesh points)
    # Shape: [batch, channels, n_points]
    x = torch.randn(batch_size, in_channels, n_points)

    # Create random positions for the point cloud (irregular mesh)
    # Shape: [batch, n_points, 2] for 2D
    pos = torch.rand(batch_size, n_points, 2)

    print(f"\nInput shape: {x.shape}")
    print(f"Position shape: {pos.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x, pos=pos)

    print(f"Output shape: {output.shape}")

    # Test with GNO2D (specialized for 2D)
    print("\n" + "=" * 60)
    print("Testing GNO2D (2D specialized)")
    print("=" * 60)

    model_2d = GNO2D(
        in_channels=in_channels,
        out_channels=out_channels,
        num_gno_layers=3,
        hidden_channels=32,
        add_grid=True,
        neighbor_strategy="knn",
        k_neighbors=8,
    )

    output_2d = model_2d(x, pos=pos)
    print(f"GNO2D output shape: {output_2d.shape}")

    # Test with radius-based neighbor search
    print("\n" + "=" * 60)
    print("Testing with radius-based neighbor search")
    print("=" * 60)

    model_radius = GNO(
        in_channels=in_channels,
        out_channels=out_channels,
        num_gno_layers=2,
        hidden_channels=32,
        neighbor_strategy="radius",
        radius=0.2,
        max_num_neighbors=16,
        add_grid=True,
    )

    output_radius = model_radius(x, pos=pos)
    print(f"Radius-based GNO output shape: {output_radius.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
