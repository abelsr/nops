"""
Tests for Graph Neural Operator (GNO) models.
"""

import pytest
import torch
import torch.nn as nn

from nops.gno.models.original import GNO, GNO2D, GNO3D
from nops.gno.layers import GNOBlock, IntegralTransform, MLP
from nops.gno.layers import radius_graph, knn_graph


class TestGNO:
    """Tests for basic GNO model."""

    def test_gno_creation(self):
        """Test GNO model creation."""
        model = GNO(
            in_channels=3,
            out_channels=1,
            num_gno_layers=2,
            hidden_channels=32,
            lifting_channels=32,
            projection_channels=32,
            add_grid=True,
            neighbor_strategy="knn",
            k_neighbors=8,
        )
        assert model is not None
        assert model.in_channels == 3
        assert model.out_channels == 1

    def test_gno_forward(self):
        """Test GNO forward pass."""
        model = GNO(
            in_channels=3,
            out_channels=1,
            num_gno_layers=2,
            hidden_channels=32,
            add_grid=True,
            neighbor_strategy="knn",
            k_neighbors=8,
        )
        model.eval()

        batch_size = 2
        n_points = 25
        x = torch.randn(batch_size, 3, n_points)
        pos = torch.rand(batch_size, n_points, 2)

        with torch.no_grad():
            output = model(x, pos=pos)

        assert output.shape == (batch_size, 1, n_points)

    def test_gno_without_positions(self):
        """Test GNO with default positions."""
        model = GNO(
            in_channels=3,
            out_channels=1,
            num_gno_layers=1,
            hidden_channels=16,
            add_grid=False,
            neighbor_strategy="knn",
            k_neighbors=4,
        )
        model.eval()

        batch_size = 1
        n_points = 16
        x = torch.randn(batch_size, 3, n_points)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 1, n_points)

    def test_gno2d(self):
        """Test GNO2D model."""
        model = GNO2D(
            in_channels=3,
            out_channels=1,
            num_gno_layers=2,
            hidden_channels=32,
            add_grid=True,
            neighbor_strategy="knn",
            k_neighbors=8,
        )

        batch_size = 2
        n_points = 25
        x = torch.randn(batch_size, 3, n_points)
        pos = torch.rand(batch_size, n_points, 2)

        with torch.no_grad():
            output = model(x, pos=pos)

        assert output.shape == (batch_size, 1, n_points)

    def test_gno_radius_strategy(self):
        """Test GNO with radius-based neighbor search."""
        model = GNO(
            in_channels=3,
            out_channels=1,
            num_gno_layers=2,
            hidden_channels=32,
            neighbor_strategy="radius",
            radius=0.3,
            max_num_neighbors=16,
            add_grid=True,
        )

        batch_size = 2
        n_points = 25
        x = torch.randn(batch_size, 3, n_points)
        pos = torch.rand(batch_size, n_points, 2)

        with torch.no_grad():
            output = model(x, pos=pos)

        assert output.shape == (batch_size, 1, n_points)


class TestNeighborSearch:
    """Tests for neighbor search functions."""

    def test_knn_graph(self):
        """Test KNN graph creation."""
        n_points = 10
        k = 3
        pos = torch.rand(n_points, 2)

        edge_index = knn_graph(pos, k=k)

        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] > 0

    def test_knn_graph_batch(self):
        """Test KNN graph with batch."""
        batch_size = 2
        n_points = 10
        k = 3
        pos = torch.rand(batch_size, n_points, 2)

        edge_index = knn_graph(pos, k=k)

        assert edge_index.shape[0] == 2

    def test_radius_graph(self):
        """Test radius graph creation."""
        n_points = 10
        r = 0.3
        pos = torch.rand(n_points, 2)

        edge_index, edge_weights = radius_graph(pos, r=r, max_num_neighbors=5)

        assert edge_index.shape[0] == 2
        # May be empty if no neighbors within radius
        assert edge_weights is not None


class TestIntegralTransform:
    """Tests for IntegralTransform layer."""

    def test_integral_transform(self):
        """Test integral transform forward."""
        layer = IntegralTransform(
            in_channels=3, out_channels=4, hidden_channels=32, n_layers=1
        )

        batch_size = 2
        n_points = 10
        n_edges = 30

        x = torch.randn(batch_size, n_points, 3)
        pos = torch.rand(batch_size, n_points, 2)
        edge_index = torch.tensor([[0, 1, 2] * 10, [1, 2, 3] * 10], dtype=torch.long)
        edge_index = edge_index[:, :n_edges]

        output = layer(x, pos, edge_index)

        assert output.shape == (batch_size, n_points, 4)


class TestMLP:
    """Tests for MLP layer."""

    def test_mlp_forward(self):
        """Test MLP forward pass."""
        mlp = MLP([8, 16, 4])

        x = torch.randn(2, 10, 8)
        output = mlp(x)

        assert output.shape == (2, 10, 4)

    def test_mlp_single(self):
        """Test MLP with single input."""
        mlp = MLP([8, 4])

        x = torch.randn(2, 8)
        output = mlp(x)

        assert output.shape == (2, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
