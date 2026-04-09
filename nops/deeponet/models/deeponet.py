import torch
import torch.nn as nn
from typing import List, Optional, Union
from nops.deeponet.layers.mlp import MLP


class DeepONet(nn.Module):
    """
    Deep Operator Network (DeepONet) for learning operators between function spaces.

    The DeepONet consists of two sub-networks:
    1. Branch network: Encodes the input function (discretized at sensor locations)
    2. Trunk network: Encodes the query locations where to evaluate the operator

    The output is computed as the dot product of the branch and trunk outputs,
    optionally followed by a transformation layer.

    Args:
        branch_layers (List[int]): Layer sizes for the branch network.
                                   Format: [input_size, hidden_size, ..., hidden_size, output_size]
        trunk_layers (List[int]): Layer sizes for the trunk network.
                                  Format: [input_size, hidden_size, ..., hidden_size, output_size]
        activation (nn.Module): Activation function to use. Default: GELU.
        num_outputs (int): Number of output functions. Default: 1.
        use_bias (bool): Whether to use bias in the final transformation. Default: True.
        dropout_rate (float): Dropout rate for regularization. Default: 0.0.
    """

    def __init__(
        self,
        branch_layers: List[int],
        trunk_layers: List[int],
        activation: nn.Module = nn.GELU(),
        num_outputs: int = 1,
        use_bias: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.branch_layers = branch_layers
        self.trunk_layers = trunk_layers
        self.num_outputs = num_outputs
        self.use_bias = use_bias

        # Ensure the output dimensions of branch and trunk match
        self.branch_output_size = branch_layers[-1]
        self.trunk_output_size = trunk_layers[-1]

        if self.branch_output_size != self.trunk_output_size:
            raise ValueError(
                f"Branch output size ({self.branch_output_size}) must match "
                f"trunk output size ({self.trunk_output_size}). "
                f"Consider using projection layers or matching the final layer sizes."
            )

        self.latent_size = self.branch_output_size

        # Build branch network
        self.branch_net = self._build_network(branch_layers, activation, dropout_rate)

        # Build trunk network
        self.trunk_net = self._build_network(trunk_layers, activation, dropout_rate)

        # Final transformation layer (optional) - only for multiple outputs
        if num_outputs > 1:
            self.output_transform = nn.Linear(
                self.latent_size, num_outputs, bias=use_bias
            )
        else:
            self.output_transform = None

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None

    def _build_network(
        self, layers: List[int], activation: nn.Module, dropout_rate: float
    ) -> nn.Sequential:
        """Build a neural network with given layer sizes."""
        network_layers = []

        for i in range(len(layers) - 1):
            network_layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:  # No activation after last layer
                network_layers.append(activation)
                if dropout_rate > 0.0:
                    network_layers.append(nn.Dropout(dropout_rate))

        return nn.Sequential(*network_layers)

    def forward(
        self, branch_input: torch.Tensor, trunk_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the DeepONet.

        Args:
            branch_input (torch.Tensor): Input function values at sensor locations.
                                       Shape: [batch_size, num_sensors, branch_input_dim]
                                       or [batch_size, branch_input_dim] for single sensor
            trunk_input (torch.Tensor): Query locations where to evaluate the operator.
                                      Shape: [batch_size, num_query_points, trunk_input_dim]
                                      or [batch_size, trunk_input_dim] for single query point

        Returns:
            torch.Tensor: Output values G(u)(y) for each query point.
                         Shape: [batch_size, num_query_points, num_outputs]
        """
        # Handle branch input
        if branch_input.dim() == 2:
            # [batch_size, branch_input_dim] -> [batch_size, 1, branch_input_dim]
            branch_input = branch_input.unsqueeze(1)

        batch_size_branch, num_sensors, branch_input_dim = branch_input.shape

        # Process branch input through branch network
        # The branch network processes the entire function (all sensor values concatenated)
        branch_flat = branch_input.view(
            batch_size_branch, -1
        )  # [batch_size, num_sensors * branch_input_dim]
        branch_output = self.branch_net(branch_flat)  # [batch_size, latent_size]

        # Handle trunk input
        if trunk_input.dim() == 2:
            # [batch_size, num_query_points] -> [batch_size, num_query_points, 1]
            # Assume trunk_input_dim = 1 for coordinate inputs
            trunk_input = trunk_input.unsqueeze(-1)

        batch_size_trunk, num_query_points, trunk_input_dim = trunk_input.shape

        # Ensure batch sizes match
        if batch_size_branch != batch_size_trunk:
            raise ValueError(
                f"Batch size mismatch: branch ({batch_size_branch}) != trunk ({batch_size_trunk})"
            )

        # Process trunk input through trunk network
        trunk_flat = trunk_input.view(
            -1, trunk_input_dim
        )  # [batch_size * num_query_points, trunk_input_dim]
        trunk_output = self.trunk_net(
            trunk_flat
        )  # [batch_size * num_query_points, latent_size]
        trunk_output = trunk_output.view(
            batch_size_trunk, num_query_points, self.latent_size
        )  # [batch_size, num_query_points, latent_size]

        # Compute element-wise product and sum (dot product in latent space)
        # branch_output: [batch_size, latent_size]
        # trunk_output: [batch_size, num_query_points, latent_size]

        # Expand branch_output to match trunk_output dimensions for broadcasting
        branch_expanded = branch_output.unsqueeze(1)  # [batch_size, 1, latent_size]

        # Element-wise multiplication
        product = (
            branch_expanded * trunk_output
        )  # [batch_size, num_query_points, latent_size]

        # Sum over latent dimension to get dot product
        dot_product = torch.sum(product, dim=-1)  # [batch_size, num_query_points]

        # Apply output transformation if needed
        if self.num_outputs > 1 and self.output_transform is not None:
            # Apply transformation for each query point
            # product: [batch_size, num_query_points, latent_size]
            batch_size, num_query_points, latent_size = product.shape
            product_flat = product.view(
                -1, latent_size
            )  # [batch_size * num_query_points, latent_size]
            output_flat = self.output_transform(
                product_flat
            )  # [batch_size * num_query_points, num_outputs]
            output = output_flat.view(batch_size, num_query_points, self.num_outputs)
        else:
            # For single output without transformation, keep as is
            output = dot_product.unsqueeze(-1)  # [batch_size, num_query_points, 1]

        # Apply dropout if specified
        if self.dropout is not None:
            output = self.dropout(output)

        return output

    def encode_branch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input function using the branch network.

        Args:
            x (torch.Tensor): Input function values at sensor locations.
                            Shape: [batch_size, num_sensors, branch_input_dim]
                            or [batch_size, branch_input_dim]

        Returns:
            torch.Tensor: Encoded branch representation.
                         Shape: [batch_size, latent_size]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size, num_sensors, input_dim = x.shape
        x_flat = x.view(batch_size, -1)  # [batch_size, num_sensors * input_dim]
        return self.branch_net(x_flat)

    def encode_trunk(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode query locations using the trunk network.

        Args:
            x (torch.Tensor): Query locations.
                            Shape: [batch_size, num_points, trunk_input_dim]
                            or [batch_size, trunk_input_dim]

        Returns:
            torch.Tensor: Encoded trunk representation.
                         Shape: [batch_size, num_points, latent_size]
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        batch_size, num_points, input_dim = x.shape
        x_flat = x.view(-1, input_dim)
        trunk_output = self.trunk_net(x_flat)
        return trunk_output.view(batch_size, num_points, self.latent_size)


class DeepONetCartesianProd(DeepONet):
    """
    DeepONet for Cartesian product formatted data.

    This variant expects inputs where:
    - branch_input: [batch_size, branch_input_dim] (same for all trunk points)
    - trunk_input: [batch_size * num_trunk_points, trunk_input_dim]

    and produces output of shape [batch_size, num_trunk_points, num_outputs].
    """

    def forward(
        self, branch_input: torch.Tensor, trunk_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for Cartesian product formatted data.

        Args:
            branch_input (torch.Tensor): Input function values.
                                       Shape: [batch_size, branch_input_dim]
            trunk_input (torch.Tensor): Query locations.
                                      Shape: [batch_size * num_trunk_points, trunk_input_dim]

        Returns:
            torch.Tensor: Output values.
                         Shape: [batch_size, num_trunk_points, num_outputs]
        """
        batch_size = branch_input.shape[0]

        # Process branch input (same for all trunk points)
        branch_output = self.branch_net(branch_input)  # [batch_size, latent_size]

        # Process trunk input
        trunk_output = self.trunk_net(
            trunk_input
        )  # [batch_size * num_trunk_points, latent_size]
        trunk_output = trunk_output.view(
            batch_size, -1, self.latent_size
        )  # [batch_size, num_trunk_points, latent_size]

        # Compute dot product for each trunk point
        # branch_output: [batch_size, latent_size] -> [batch_size, 1, latent_size]
        # trunk_output: [batch_size, num_trunk_points, latent_size]
        branch_expanded = branch_output.unsqueeze(1)
        product = (
            branch_expanded * trunk_output
        )  # [batch_size, num_trunk_points, latent_size]
        dot_product = torch.sum(product, dim=-1)  # [batch_size, num_trunk_points]

        # Apply output transformation
        if self.num_outputs > 1 or self.output_transform is not None:
            output = self.output_transform(
                dot_product.unsqueeze(-1)
            )  # [batch_size, num_trunk_points, num_outputs]
        else:
            output = dot_product.unsqueeze(-1)  # [batch_size, num_trunk_points, 1]

        # Apply dropout if specified
        if self.dropout is not None:
            output = self.dropout(output)

        return output
