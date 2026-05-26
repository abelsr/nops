import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from vlasov_poisson.utils.wandb_utils import log_artifact

console = Console()


def _format_progress_metrics(train_rel=None, train_mse=None, test_rel=None, test_mse=None, lr=None):
    train_rel = "--" if train_rel is None else f"{train_rel:.4f}"
    train_mse = "--" if train_mse is None else f"{train_mse:.6f}"
    test_rel = "--" if test_rel is None else f"{test_rel:.4f}"
    test_mse = "--" if test_mse is None else f"{test_mse:.6f}"
    lr = "--" if lr is None else f"{lr:.2e}"
    return f"train rel_l2={train_rel} mse={train_mse} | test rel_l2={test_rel} mse={test_mse} | lr={lr}"


def train_epoch(model, train_loader, optimizer, criterion_rel, criterion_mse, device, progress, task_id):
    model.train()
    train_loss_rel = 0.0
    train_loss_mse = 0.0
    seen = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        pred = model(x)

        loss_rel = criterion_rel(pred, y)
        loss_mse = criterion_mse(pred, y)
        loss = loss_rel + 0.1 * loss_mse

        loss.backward()
        optimizer.step()

        train_loss_rel += loss_rel.item() * x.size(0)
        train_loss_mse += loss_mse.item() * x.size(0)
        seen += x.size(0)

        progress.update(
            task_id,
            advance=1,
            phase="train",
            metrics=_format_progress_metrics(
                train_rel=train_loss_rel / seen,
                train_mse=train_loss_mse / seen,
                lr=optimizer.param_groups[0]["lr"],
            ),
        )

    train_loss_rel /= len(train_loader.dataset)
    train_loss_mse /= len(train_loader.dataset)
    return train_loss_rel, train_loss_mse


def validate(model, test_loader, criterion_rel, criterion_mse, device, progress, task_id, train_loss_rel, train_loss_mse, optimizer):
    model.eval()
    val_loss_rel = 0.0
    val_loss_mse = 0.0
    seen = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            loss_rel = criterion_rel(pred, y)
            loss_mse = criterion_mse(pred, y)

            val_loss_rel += loss_rel.item() * x.size(0)
            val_loss_mse += loss_mse.item() * x.size(0)
            seen += x.size(0)

            progress.update(
                task_id,
                advance=1,
                phase="test",
                metrics=_format_progress_metrics(
                    train_rel=train_loss_rel,
                    train_mse=train_loss_mse,
                    test_rel=val_loss_rel / seen,
                    test_mse=val_loss_mse / seen,
                    lr=optimizer.param_groups[0]["lr"],
                ),
            )

    val_loss_rel /= len(test_loader.dataset)
    val_loss_mse /= len(test_loader.dataset)
    return val_loss_rel, val_loss_mse


def train(model, train_loader, test_loader, optimizer, scheduler, criterion_rel, criterion_mse, epochs, device, results_dir, run=None):
    if run is not None:
        run.config.update({"optimizer": "AdamW", "scheduler": "CosineAnnealingLR"}, allow_val_change=True)

    for epoch in range(epochs):
        t_epoch_start = time.time()

        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            TextColumn("[{task.fields[phase]}]"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.fields[metrics]}"),
            console=console,
            transient=True,
        )
        with progress:
            task_id = progress.add_task(
                f"Epoch {epoch + 1:03d}/{epochs:03d}",
                total=len(train_loader) + len(test_loader),
                phase="train",
                metrics=_format_progress_metrics(lr=optimizer.param_groups[0]["lr"]),
            )
            train_loss_rel, train_loss_mse = train_epoch(
                model, train_loader, optimizer, criterion_rel, criterion_mse, device, progress, task_id
            )
            val_loss_rel, val_loss_mse = validate(
                model,
                test_loader,
                criterion_rel,
                criterion_mse,
                device,
                progress,
                task_id,
                train_loss_rel,
                train_loss_mse,
                optimizer,
            )
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch + 1,
            "train/rel_l2": train_loss_rel,
            "train/mse": train_loss_mse,
            "test/rel_l2": val_loss_rel,
            "test/mse": val_loss_mse,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": time.time() - t_epoch_start,
        }
        if run is not None:
            run.log(epoch_metrics, step=epoch + 1)

        console.print(
            f"Epoch {epoch + 1:03d}/{epochs:03d} | "
            f"train rel_l2={train_loss_rel:.4f} mse={train_loss_mse:.6f} | "
            f"test rel_l2={val_loss_rel:.4f} mse={val_loss_mse:.6f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"time={epoch_metrics['epoch_time_sec']:.1f}s"
        )

    model_path = results_dir / "fno3d_checkpoint.pt"
    torch.save(model.state_dict(), model_path)
    log_artifact(model_path, "model")
    print(f"\nSaved model checkpoint to: {model_path}")

    return val_loss_rel, val_loss_mse


def evaluate_gabriela(model, data_dir, device):
    print("\n=========================================================")
    print("Evaluating on GABRIELA dataset (256^3 -> 64^3 -> 256^3)...")
    print("=========================================================")

    g_dir = data_dir / "gabriela" / "density"
    g_files = sorted(list(g_dir.glob("*.npy")), key=lambda f: float(f.stem.split("_")[1]), reverse=True)
    g_zs = [float(f.stem.split("_")[1]) for f in g_files]
    g_as = torch.tensor([1.0 / (1.0 + z) for z in g_zs], dtype=torch.float32)

    g_pairs = []
    pairs_file = g_dir / "pairs.csv"
    if pairs_file.exists():
        pairs_df = np.genfromtxt(pairs_file, delimiter=',', dtype=str, skip_header=1)
        if pairs_df.ndim == 1:
            pairs_df = np.expand_dims(pairs_df, axis=0)
        for row in pairs_df:
            g_pairs.append((float(row[0]), float(row[1]), row[2], row[3]))

    if not g_pairs:
        print("  Warning: No pairs.csv found or empty. Using adjacent files as pairs.")
        for idx in range(len(g_files) - 1):
            z0 = float(g_files[idx].stem.split("_")[1])
            z1 = float(g_files[idx+1].stem.split("_")[1])
            g_pairs.append((z0, z1, g_files[idx].name, g_files[idx+1].name))

    print(f"  Evaluating {len(g_pairs)} transition pairs...")

    model.eval()
    gabriela_errors = []
    plot_pair = g_pairs[len(g_pairs) // 2] if g_pairs else None
    plot_data = None

    with torch.no_grad():
        for pair_idx, (z0, z1, f_z0, f_z1) in enumerate(g_pairs):
            rho256_in = torch.from_numpy(np.load(g_dir / f_z0).astype(np.float32))
            rho256_out = torch.from_numpy(np.load(g_dir / f_z1).astype(np.float32))

            a0 = 1.0 / (1.0 + z0)
            a1 = 1.0 / (1.0 + z1)
            delta_a = a1 - a0

            rho64_in = F.avg_pool3d(rho256_in.unsqueeze(0).unsqueeze(1), kernel_size=4, stride=4).squeeze(0).squeeze(0)

            delta_a_channel = torch.full_like(rho64_in, delta_a)
            x_in = torch.stack([rho64_in, delta_a_channel], dim=0).unsqueeze(0).to(device)

            pred64 = model(x_in).cpu().squeeze(0).squeeze(0)

            pred256 = F.interpolate(pred64.unsqueeze(0).unsqueeze(0), size=(256, 256, 256), mode="trilinear", align_corners=True).squeeze(0).squeeze(0)

            diff_norm = torch.linalg.vector_norm(pred256 - rho256_out, ord=2)
            target_norm = torch.linalg.vector_norm(rho256_out, ord=2)
            rel_error = (diff_norm / (target_norm + 1e-8)).item()
            gabriela_errors.append(rel_error)

            print(f"  Pair {pair_idx+1:02d}/{len(g_pairs)}: z={z0:.2f}->{z1:.2f} (delta_a={delta_a:.4f}) | Rel L2 Error: {rel_error:.4f}")

            if plot_pair and z0 == plot_pair[0] and z1 == plot_pair[1]:
                plot_data = {
                    "in_slice": rho256_in[:, :, 128].numpy(),
                    "target_slice": rho256_out[:, :, 128].numpy(),
                    "pred_slice": pred256[:, :, 128].numpy(),
                    "z0": z0,
                    "z1": z1
                }

    mean_g_error = np.mean(gabriela_errors)
    print(f"\nMean Relative L2 Error on Gabriela Dataset: {mean_g_error:.4f}")
    if wandb.run is not None:
        wandb.log({"gabriela/mean_rel_l2": mean_g_error})
    return gabriela_errors, plot_data, mean_g_error


def plot_gabriela_comparison(plot_data, results_dir):
    if not plot_data:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im0 = axes[0].imshow(plot_data["in_slice"], cmap="viridis", origin="lower")
    axes[0].set_title(f"Input z={plot_data['z0']:.2f} (256^3)")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(plot_data["target_slice"], cmap="viridis", origin="lower")
    axes[1].set_title(f"Target z={plot_data['z1']:.2f} (256^3)")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(plot_data["pred_slice"], cmap="viridis", origin="lower")
    axes[2].set_title("FNO3D Prediction (Upsampled)")
    fig.colorbar(im2, ax=axes[2])

    plt.suptitle(f"Gabriela Test Slice Comparison (Redshift z={plot_data['z0']:.2f} -> z={plot_data['z1']:.2f})", fontsize=14)
    plt.tight_layout()
    plot_path = results_dir / "gabriela_fno_prediction.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved Gabriela prediction plot to: {plot_path}")


def plot_miguel_comparison(model, test_dataset, device, results_dir):
    model.eval()
    with torch.no_grad():
        x_sample, y_sample = test_dataset[len(test_dataset) // 2]
        x_in = x_sample.unsqueeze(0).to(device)
        pred_sample = model(x_in).cpu().squeeze(0).squeeze(0)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

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


def save_metrics(val_loss_rel, val_loss_mse, gabriela_errors, mean_g_error, results_dir):
    metrics = {
        "miguel_64_test_val_rel_l2": val_loss_rel,
        "miguel_64_test_val_mse": val_loss_mse,
        "gabriela_test_mean_rel_l2": mean_g_error,
        "gabriela_pair_errors": gabriela_errors
    }
    with open(results_dir / "fno3d_evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Saved evaluation metrics JSON.")
