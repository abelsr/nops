import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        num_layers: int = 2,
        activation: nn.Module = nn.GELU(),
    ):
        """
        Multi-Layer Perceptron (MLP) module for DeepONet.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            hidden_features (int): Number of features in hidden layers.
            num_layers (int): Number of hidden layers. Default: 2.
            activation (nn.Module): Activation function. Default: GELU.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.activation = activation

        # Build layers
        layers = []
        # Input layer
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(activation)

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(activation)

        # Output layer
        layers.append(nn.Linear(hidden_features, out_features))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (*, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (*, out_features).
        """
        return self.mlp(x)
