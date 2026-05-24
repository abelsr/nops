import os
import time
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Import FNO from nops
from nops.fno.models.original import FNO

# ---------------------------------------------------------
# 1. Custom Dataset for Variable Step Size (delta_a)
# ---------------------------------------------------------
class VlasovPoissonDataset(Dataset):
    def __init__(self, data_tensor, a_values, is_train=True, max_step=100, num_samples=10000):
        """
        Args:
            data_tensor (torch.Tensor): Tensor of shape (N, 64, 64, 64) containing simulation density snapshots.
            a_values (torch.Tensor): Tensor of shape (N,) containing scale factor values for each snapshot.
            is_train (bool): If True, randomly samples pairs (t_i, t_j) for training.
            max_step (int): Maximum snapshot index difference for a prediction step.
            num_samples (int): Number of random samples per epoch (for training).
        """
        self.data_tensor = data_tensor
        self.a_values = a_values
        self.is_train = is_train
        self.max_step = max_step
        self.num_samples = num_samples
        self.N = data_tensor.shape[0]

    def __len__(self):
        if self.is_train:
            return self.num_samples
        else:
            # For validation, we evaluate step predictions starting at each possible index
            return max(1, self.N - 20)

    def __getitem__(self, idx):
        if self.is_train:
            # Randomly select start index i
            i = np.random.randint(0, self.N - 1)
            # Randomly select end index j in [i+1, min(i+max_step, N-1)]
            j = np.random.randint(i + 1, min(i + self.max_step + 1, self.N))
        else:
            # Deterministic validation: predict a step of size max_step // 2
            i = idx
            j = min(i + max(1, self.max_step // 2), self.N - 1)
            if j <= i:
                j = min(i + 1, self.N - 1)

        rho_in = self.data_tensor[i]   # (64, 64, 64)
        rho_out = self.data_tensor[j]  # (64, 64, 64)

        delta_a = self.a_values[j] - self.a_values[i]

        # Input tensor has two channels: 
        # Channel 0: density map rho_in
        # Channel 1: constant delta_a parameter
        delta_a_channel = torch.full_like(rho_in, delta_a)
        x = torch.stack([rho_in, delta_a_channel], dim=0)  # (2, 64, 64, 64)
        y = rho_out.unsqueeze(0)                           # (1, 64, 64, 64)

        return x, y

# ---------------------------------------------------------
# 2. Relative L2 Loss (standard for FNO training)
# ---------------------------------------------------------
class RelativeL2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Shape: (B, C, H, W, D)
        # Compute L2 norm over spatial dimensions (H, W, D)
        diff_norm = torch.linalg.vector_norm(pred - target, ord=2, dim=(2, 3, 4))
        target_norm = torch.linalg.vector_norm(target, ord=2, dim=(2, 3, 4))
        # Avoid division by zero
        relative_error = diff_norm / (target_norm + 1e-8)
        return torch.mean(relative_error)

# ---------------------------------------------------------
# 3. Main Training & Evaluation Pipeline
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train FNO3D on Vlasov-Poisson cosmological simulations.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=type(1e-3), default=1e-3, help="Learning rate")
    parser.add_argument("--max-step", type=int, default=50, help="Maximum snapshot index difference for a pair")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick dry run to verify the training pipeline")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up directories
    base_dir = Path("experiments/vlasov-poisson")
    data_dir = base_dir / "data" / "simulations"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # ---------------------------------------------------------
    # A. Load Miguel_64 dataset
    # ---------------------------------------------------------
    print("Loading Miguel_64 dataset...")
    m_dir = data_dir / "miguel_64" / "density"
    m_files = sorted(list(m_dir.glob("*.npy")), key=lambda f: float(f.stem.split("_")[1]), reverse=True)
    m_zs = [float(f.stem.split("_")[1]) for f in m_files]
    m_as = torch.tensor([1.0 / (1.0 + z) for z in m_zs], dtype=torch.float32)
    
    if args.dry_run:
        m_files = m_files[:100]
        m_as = m_as[:100]
    
    print(f"  Loaded {len(m_files)} snapshots of size (64, 64, 64)")
    
    # Load all snapshots into a single tensor
    snapshots = []
    t0 = time.time()
    for f in m_files:
        snapshots.append(torch.from_numpy(np.load(f).astype(np.float32)))
    m_data = torch.stack(snapshots, dim=0) # (N, 64, 64, 64)
    print(f"  Finished loading in {time.time() - t0:.2f}s. Shape: {m_data.shape}")

    # Split into train (first 80% of snapshots) and test (last 20% of snapshots)
    num_snapshots = len(m_files)
    train_split = int(num_snapshots * 0.8)
    
    m_train_data = m_data[:train_split]
    m_train_as = m_as[:train_split]
    
    m_test_data = m_data[train_split:]
    m_test_as = m_as[train_split:]
    
    print(f"  Train snapshots: {m_train_data.shape[0]} | Test snapshots: {m_test_data.shape[0]}")

    # Create datasets
    train_dataset = VlasovPoissonDataset(
        m_train_data, m_train_as, is_train=True, 
        max_step=args.max_step, num_samples=1000 if args.dry_run else 5000
    )
    test_dataset = VlasovPoissonDataset(
        m_test_data, m_test_as, is_train=False, 
        max_step=args.max_step
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ---------------------------------------------------------
    # B. Model Setup: FNO3D
    # ---------------------------------------------------------
    print("\nInstantiating FNO3D model...")
    # Modes represents number of Fourier modes in each spatial dimension (X, Y, Z)
    model = FNO(
        modes=[8, 8, 8],
        num_fourier_layers=3,
        in_channels=2,          # channel 0: density, channel 1: delta_a
        lifting_channels=16,
        projection_channels=16,
        out_channels=1,         # output density map
        mid_channels=32,
        activation=nn.GELU(),
        add_grid=True,          # helps model identify absolute grid coords
        n_fno_blocks_per_layer=1,
        dropout=0.05
    )
    model = model.to(device)
    
    # Compute parameter count
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  FNO3D parameter count: {params:,}")

    # Losses & Optimizer
    criterion_rel = RelativeL2Loss()
    criterion_mse = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---------------------------------------------------------
    # C. Training Loop
    # ---------------------------------------------------------
    epochs = 2 if args.dry_run else args.epochs
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        train_loss_rel = 0.0
        train_loss_mse = 0.0
        t_epoch_start = time.time()
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            pred = model(x)
            
            # Combine MSE and Relative L2 loss
            loss_rel = criterion_rel(pred, y)
            loss_mse = criterion_mse(pred, y)
            loss = loss_rel + 0.1 * loss_mse
            
            loss.backward()
            optimizer.step()
            
            train_loss_rel += loss_rel.item() * x.size(0)
            train_loss_mse += loss_mse.item() * x.size(0)
            
        train_loss_rel /= len(train_loader.dataset)
        train_loss_mse /= len(train_loader.dataset)
        scheduler.step()
        
        # Validation on test set (Miguel_64)
        model.eval()
        val_loss_rel = 0.0
        val_loss_mse = 0.0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                
                loss_rel = criterion_rel(pred, y)
                loss_mse = criterion_mse(pred, y)
                
                val_loss_rel += loss_rel.item() * x.size(0)
                val_loss_mse += loss_mse.item() * x.size(0)
                
        val_loss_rel /= len(test_loader.dataset)
        val_loss_mse /= len(test_loader.dataset)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train RelL2: {train_loss_rel:.4f}, MSE: {train_loss_mse:.6f} | "
              f"Val RelL2: {val_loss_rel:.4f}, MSE: {val_loss_mse:.6f} | Time: {time.time() - t_epoch_start:.1f}s")

    # Save model checkpoint
    model_path = results_dir / "fno3d_checkpoint.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved model checkpoint to: {model_path}")

    # ---------------------------------------------------------
    # D. Evaluate on Gabriela's Dataset (256^3) - Anti-OOM Pipeline
    # ---------------------------------------------------------
    print("\n=========================================================")
    print("Evaluating on GABRIELA dataset (256^3 -> 64^3 -> 256^3)...")
    print("=========================================================")
    
    g_dir = data_dir / "gabriela" / "density"
    g_files = sorted(list(g_dir.glob("*.npy")), key=lambda f: float(f.stem.split("_")[1]), reverse=True)
    g_zs = [float(f.stem.split("_")[1]) for f in g_files]
    g_as = torch.tensor([1.0 / (1.0 + z) for z in g_zs], dtype=torch.float32)

    # Read pairs.csv for Gabriela's dataset
    g_pairs = []
    pairs_file = g_dir / "pairs.csv"
    if pairs_file.exists():
        pairs_df = pd_df = np.genfromtxt(pairs_file, delimiter=',', dtype=str, skip_header=1)
        # Handle cases where file has 1 row or many
        if pairs_df.ndim == 1:
            pairs_df = np.expand_dims(pairs_df, axis=0)
        for row in pairs_df:
            g_pairs.append((float(row[0]), float(row[1]), row[2], row[3]))
    
    if not g_pairs:
        # Fallback if no pairs file: define step size of 1 snapshot indices
        print("  Warning: No pairs.csv found or empty. Using adjacent files as pairs.")
        for idx in range(len(g_files) - 1):
            z0 = float(g_files[idx].stem.split("_")[1])
            z1 = float(g_files[idx+1].stem.split("_")[1])
            g_pairs.append((z0, z1, g_files[idx].name, g_files[idx+1].name))
            
    print(f"  Evaluating {len(g_pairs)} transition pairs...")
    
    model.eval()
    gabriela_errors = []
    
    # Track one pair to plot predictions
    plot_pair = g_pairs[len(g_pairs) // 2] if g_pairs else None
    plot_data = None
    
    with torch.no_grad():
        for pair_idx, (z0, z1, f_z0, f_z1) in enumerate(g_pairs):
            # Load high-res snapshots in CPU (float32 for efficiency)
            rho256_in = torch.from_numpy(np.load(g_dir / f_z0).astype(np.float32))
            rho256_out = torch.from_numpy(np.load(g_dir / f_z1).astype(np.float32))
            
            a0 = 1.0 / (1.0 + z0)
            a1 = 1.0 / (1.0 + z1)
            delta_a = a1 - a0
            
            # --- 1. Downsample (256^3 -> 64^3) on CPU using 3D average pooling ---
            #avg_pool3d expects (B, C, D, H, W)
            rho64_in = F.avg_pool3d(rho256_in.unsqueeze(0).unsqueeze(0), kernel_size=4, stride=4).squeeze(0).squeeze(0)
            
            # --- 2. Inference on GPU at 64^3 ---
            delta_a_channel = torch.full_like(rho64_in, delta_a)
            x_in = torch.stack([rho64_in, delta_a_channel], dim=0).unsqueeze(0).to(device) # (1, 2, 64, 64, 64)
            
            pred64 = model(x_in).cpu().squeeze(0).squeeze(0) # (64, 64, 64)
            
            # --- 3. Upsample (64^3 -> 256^3) on CPU using Trilinear interpolation ---
            pred256 = F.interpolate(pred64.unsqueeze(0).unsqueeze(0), size=(256, 256, 256), mode="trilinear", align_corners=True).squeeze(0).squeeze(0)
            
            # --- 4. Compute Relative L2 error at 256^3 ---
            diff_norm = torch.linalg.vector_norm(pred256 - rho256_out, ord=2)
            target_norm = torch.linalg.vector_norm(rho256_out, ord=2)
            rel_error = (diff_norm / (target_norm + 1e-8)).item()
            gabriela_errors.append(rel_error)
            
            print(f"  Pair {pair_idx+1:02d}/{len(g_pairs)}: z={z0:.2f}->{z1:.2f} (delta_a={delta_a:.4f}) | Rel L2 Error: {rel_error:.4f}")
            
            # Cache data for plotting
            if plot_pair and z0 == plot_pair[0] and z1 == plot_pair[1]:
                # Slice through center (z=128 for 256^3, z=32 for 64^3)
                plot_data = {
                    "in_slice": rho256_in[:, :, 128].numpy(),
                    "target_slice": rho256_out[:, :, 128].numpy(),
                    "pred_slice": pred256[:, :, 128].numpy(),
                    "z0": z0,
                    "z1": z1
                }
                
    mean_g_error = np.mean(gabriela_errors)
    print(f"\nMean Relative L2 Error on Gabriela Dataset: {mean_g_error:.4f}")

    # ---------------------------------------------------------
    # E. Plot and Save Visual Predictions
    # ---------------------------------------------------------
    # Plot Gabriela slice comparison
    if plot_data:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        
        im0 = axes[0].imshow(plot_data["in_slice"], cmap="viridis", origin="lower")
        axes[0].set_title(f"Input z={plot_data['z0']:.2f} (256^3)")
        fig.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].imshow(plot_data["target_slice"], cmap="viridis", origin="lower")
        axes[1].set_title(f"Target z={plot_data['z1']:.2f} (256^3)")
        fig.colorbar(im1, ax=axes[1])
        
        im2 = axes[2].imshow(plot_data["pred_slice"], cmap="viridis", origin="lower")
        axes[2].set_title(f"FNO3D Prediction (Upsampled)")
        fig.colorbar(im2, ax=axes[2])
        
        plt.suptitle(f"Gabriela Test Slice Comparison (Redshift z={plot_data['z0']:.2f} -> z={plot_data['z1']:.2f})", fontsize=14)
        plt.tight_layout()
        plot_path = results_dir / "gabriela_fno_prediction.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Saved Gabriela prediction plot to: {plot_path}")
        
    # Plot Miguel_64 sample comparison
    model.eval()
    with torch.no_grad():
        x_sample, y_sample = test_dataset[len(test_dataset) // 2]
        x_in = x_sample.unsqueeze(0).to(device)
        pred_sample = model(x_in).cpu().squeeze(0).squeeze(0)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        
        # Center slice (z=32)
        in_slice = x_sample[0, :, :, 32].numpy()
        target_slice = y_sample[0, :, :, 32].numpy()
        pred_slice = pred_sample[:, :, 32].numpy()
        
        im0 = axes[0].imshow(in_slice, cmap="viridis", origin="lower")
        axes[0].set_title("Input Density (64^3)")
        fig.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].imshow(target_slice, cmap="viridis", origin="lower")
        axes[1].set_title("Target Density (64^3)")
        fig.colorbar(im1, ax=axes[1])
        
        im2 = axes[2].imshow(pred_slice, cmap="viridis", origin="lower")
        axes[2].set_title("FNO3D Prediction (64^3)")
        fig.colorbar(im2, ax=axes[2])
        
        plt.suptitle("Miguel_64 Test Slice Comparison", fontsize=14)
        plt.tight_layout()
        m_plot_path = results_dir / "miguel_fno_prediction.png"
        plt.savefig(m_plot_path, dpi=150, bbox_inches="tight")
        print(f"Saved Miguel_64 prediction plot to: {m_plot_path}")

    # Write evaluation metrics to file
    metrics = {
        "miguel_64_test_val_rel_l2": val_loss_rel,
        "miguel_64_test_val_mse": val_loss_mse,
        "gabriela_test_mean_rel_l2": mean_g_error,
        "gabriela_pair_errors": gabriela_errors
    }
    with open(results_dir / "fno3d_evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Saved evaluation metrics JSON.")

if __name__ == "__main__":
    main()
