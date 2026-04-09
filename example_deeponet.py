"""
Example usage of DeepONet for learning the integration operator:
G(u)(y) = ∫_0^y u(x) dx
"""

import torch
import torch.nn as nn
from nops.deeponet.models import DeepONet


def generate_integration_data(num_samples=1000, num_sensors=100, num_query_points=50):
    """Generate training data for the integration operator."""
    # Generate random functions u(x) as combinations of sine waves
    x = torch.linspace(0, 1, num_sensors)

    branch_inputs = []
    trunk_inputs = []
    outputs = []

    for _ in range(num_samples):
        # Random coefficients for sine series
        coeffs = torch.randn(10) * 0.5

        # Generate function u(x) = Σ coeff_i * sin(i * π * x)
        u_vals = torch.zeros_like(x)
        for i in range(10):
            u_vals += coeffs[i] * torch.sin((i + 1) * torch.pi * x)

        # Generate query points y
        y = torch.linspace(0, 1, num_query_points)

        # Compute G(u)(y) = ∫_0^y u(x) dx (using trapezoidal rule)
        G_u_vals = torch.zeros_like(y)
        for i, y_val in enumerate(y):
            # Find indices where x <= y_val
            mask = x <= y_val
            if mask.sum() > 0:
                x_sub = x[mask]
                u_sub = u_vals[mask]
                # Trapezoidal integration
                G_u_vals[i] = torch.trapz(u_sub, x_sub)

        branch_inputs.append(u_vals)
        trunk_inputs.append(y)
        outputs.append(G_u_vals)

    # Stack into tensors
    branch_tensor = torch.stack(branch_inputs)  # [num_samples, num_sensors]
    trunk_tensor = torch.stack(trunk_inputs)  # [num_samples, num_query_points]
    output_tensor = torch.stack(outputs)  # [num_samples, num_query_points]

    return branch_tensor, trunk_tensor, output_tensor


def train_deeponet(model, branch_data, trunk_data, target_data, epochs=1000, lr=0.001):
    """Train the DeepONet model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        predictions = model(branch_data, trunk_data)
        loss = criterion(predictions.squeeze(-1), target_data)

        # Backward pass
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    return model


def main():
    print("Generating training data...")
    branch_data, trunk_data, target_data = generate_integration_data(
        num_samples=500, num_sensors=50, num_query_points=20
    )

    print(f"Branch data shape: {branch_data.shape}")
    print(f"Trunk data shape: {trunk_data.shape}")
    print(f"Target data shape: {target_data.shape}")

    # Create DeepONet model
    print("\nCreating DeepONet model...")
    model = DeepONet(
        branch_layers=[50, 64, 64, 32],  # [sensors, hidden, hidden, latent]
        trunk_layers=[1, 32, 32, 32],  # [coord_dim, hidden, hidden, latent]
        activation=nn.GELU(),
        num_outputs=1,
        dropout_rate=0.1,
    )

    print(f"Model architecture:\n{model}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")

    # Train the model
    print("\nTraining DeepONet...")
    trained_model = train_deeponet(
        model, branch_data, trunk_data, target_data, epochs=500
    )

    # Test the model
    print("\nTesting DeepONet...")
    trained_model.eval()
    with torch.no_grad():
        test_branch, test_trunk, test_target = generate_integration_data(
            num_samples=10, num_sensors=50, num_query_points=20
        )
        predictions = trained_model(test_branch, test_trunk)
        mse = nn.MSELoss()(predictions.squeeze(-1), test_target)
        print(f"Test MSE: {mse.item():.6f}")

        # Show some predictions vs targets
        print("\nSample predictions vs targets:")
        for i in range(3):
            print(f"Sample {i + 1}:")
            print(f"  Predictions: {predictions[i, :5].squeeze()}")
            print(f"  Targets:     {test_target[i, :5]}")
            print()


if __name__ == "__main__":
    main()
