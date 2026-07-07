import json
import time
import copy

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from vlasov_poisson.utils.wandb_utils import log_artifact

console = Console()


def predict_residual(model, x):
    rho_in = x[:, :1]
    delta_a = x[:, 1:2]
    return rho_in + delta_a * model(x)


def residual_derivative_target(x, y):
    rho_in = x[:, :1]
    delta_a = x[:, 1:2].clamp_min(1e-8)
    return (y - rho_in) / delta_a


def _relative_l2_per_sample(pred, target):
    diff_norm = torch.linalg.vector_norm(pred - target, ord=2, dim=(2, 3, 4))
    target_norm = torch.linalg.vector_norm(target, ord=2, dim=(2, 3, 4))
    return (diff_norm / (target_norm + 1e-8)).mean(dim=1)


def _delta_a_bin(delta_a):
    if delta_a < 0.005:
        return "small"
    if delta_a < 0.05:
        return "medium"
    return "large"


def _empty_bin_stats():
    return {name: {"count": 0, "model_rel_sum": 0.0, "identity_rel_sum": 0.0} for name in ("small", "medium", "large")}


def _finalize_bin_stats(bin_stats):
    finalized = {}
    for name, stats in bin_stats.items():
        count = stats["count"]
        if count == 0:
            finalized[name] = {"count": 0, "model_rel_l2": None, "identity_rel_l2": None, "model_identity_ratio": None}
            continue
        model_rel = stats["model_rel_sum"] / count
        identity_rel = stats["identity_rel_sum"] / count
        finalized[name] = {
            "count": count,
            "model_rel_l2": model_rel,
            "identity_rel_l2": identity_rel,
            "model_identity_ratio": model_rel / (identity_rel + 1e-12),
        }
    return finalized


def _update_bin_stats(bin_stats, model_rel_samples, identity_rel_samples, delta_a_samples):
    for model_rel, base_rel, delta_a in zip(model_rel_samples, identity_rel_samples, delta_a_samples):
        stats = bin_stats[_delta_a_bin(float(delta_a))]
        stats["count"] += 1
        stats["model_rel_sum"] += float(model_rel)
        stats["identity_rel_sum"] += float(base_rel)


def _format_progress_metrics(train_rel=None, train_mse=None, test_rel=None, test_mse=None, lr=None):
    train_rel = "--" if train_rel is None else f"{train_rel:.4f}"
    train_mse = "--" if train_mse is None else f"{train_mse:.6f}"
    test_rel = "--" if test_rel is None else f"{test_rel:.4f}"
    test_mse = "--" if test_mse is None else f"{test_mse:.6f}"
    lr = "--" if lr is None else f"{lr:.2e}"
    return f"train rel_l2={train_rel} mse={train_mse} | test rel_l2={test_rel} mse={test_mse} | lr={lr}"


def train_epoch(model, train_loader, optimizer, criterion_rel, criterion_mse, device, progress, task_id, train_on_residual_derivative=False):
    model.train()
    train_loss_rel = 0.0
    train_loss_mse = 0.0
    seen = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        correction = model(x)
        pred = x[:, :1] + x[:, 1:2] * correction

        pred_rel = criterion_rel(pred, y)
        pred_mse = criterion_mse(pred, y)
        if train_on_residual_derivative:
            target_correction = residual_derivative_target(x, y)
            loss_rel = criterion_rel(correction, target_correction)
            loss_mse = criterion_mse(correction, target_correction)
        else:
            loss_rel = pred_rel
            loss_mse = pred_mse
        loss = loss_rel + 0.1 * loss_mse

        loss.backward()
        optimizer.step()

        train_loss_rel += pred_rel.item() * x.size(0)
        train_loss_mse += pred_mse.item() * x.size(0)
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
    identity_loss_rel = 0.0
    identity_loss_mse = 0.0
    bin_stats = _empty_bin_stats()
    seen = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = predict_residual(model, x)
            identity = x[:, :1]

            loss_rel = criterion_rel(pred, y)
            loss_mse = criterion_mse(pred, y)
            identity_rel = criterion_rel(identity, y)
            identity_mse = criterion_mse(identity, y)

            val_loss_rel += loss_rel.item() * x.size(0)
            val_loss_mse += loss_mse.item() * x.size(0)
            identity_loss_rel += identity_rel.item() * x.size(0)
            identity_loss_mse += identity_mse.item() * x.size(0)

            model_rel_samples = _relative_l2_per_sample(pred, y).detach().cpu()
            identity_rel_samples = _relative_l2_per_sample(identity, y).detach().cpu()
            delta_a_samples = x[:, 1, 0, 0, 0].detach().cpu()
            _update_bin_stats(bin_stats, model_rel_samples, identity_rel_samples, delta_a_samples)
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
    identity_loss_rel /= len(test_loader.dataset)
    identity_loss_mse /= len(test_loader.dataset)
    diagnostics = {
        "identity_rel_l2": identity_loss_rel,
        "identity_mse": identity_loss_mse,
        "model_identity_ratio": val_loss_rel / (identity_loss_rel + 1e-12),
        "delta_a_bins": _finalize_bin_stats(bin_stats),
    }
    return val_loss_rel, val_loss_mse, diagnostics


def validate_sampled_pairs(model, dataset, criterion_rel, criterion_mse, device, batch_size, num_samples=512):
    model.eval()
    rng = np.random.RandomState(12345)
    rel_sum = 0.0
    mse_sum = 0.0
    identity_rel_sum = 0.0
    identity_mse_sum = 0.0
    bin_stats = _empty_bin_stats()
    seen = 0

    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - start)
            xs = []
            ys = []
            for _ in range(current_batch_size):
                i, j = dataset.sample_pair_indices(rng)
                x, y = dataset.make_pair(i, j)
                xs.append(x)
                ys.append(y)

            x = torch.stack(xs, dim=0).to(device)
            y = torch.stack(ys, dim=0).to(device)
            pred = predict_residual(model, x)
            identity = x[:, :1]

            rel = criterion_rel(pred, y)
            mse = criterion_mse(pred, y)
            identity_rel = criterion_rel(identity, y)
            identity_mse = criterion_mse(identity, y)

            rel_sum += rel.item() * x.size(0)
            mse_sum += mse.item() * x.size(0)
            identity_rel_sum += identity_rel.item() * x.size(0)
            identity_mse_sum += identity_mse.item() * x.size(0)

            model_rel_samples = _relative_l2_per_sample(pred, y).detach().cpu()
            identity_rel_samples = _relative_l2_per_sample(identity, y).detach().cpu()
            delta_a_samples = x[:, 1, 0, 0, 0].detach().cpu()
            _update_bin_stats(bin_stats, model_rel_samples, identity_rel_samples, delta_a_samples)
            seen += x.size(0)

    rel_l2 = rel_sum / seen
    mse = mse_sum / seen
    identity_rel_l2 = identity_rel_sum / seen
    identity_mse = identity_mse_sum / seen
    return {
        "num_samples": seen,
        "rel_l2": rel_l2,
        "mse": mse,
        "identity_rel_l2": identity_rel_l2,
        "identity_mse": identity_mse,
        "model_identity_ratio": rel_l2 / (identity_rel_l2 + 1e-12),
        "delta_a_bins": _finalize_bin_stats(bin_stats),
    }


def train(model, train_loader, test_loader, optimizer, scheduler, criterion_rel, criterion_mse, epochs, device, results_dir, run=None, train_on_residual_derivative=False):
    if run is not None:
        run.config.update({
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "train_on_residual_derivative": train_on_residual_derivative,
        }, allow_val_change=True)

    best_ratio = float("inf")
    best_epoch = None
    best_val_loss_rel = None
    best_val_loss_mse = None
    best_val_diagnostics = None
    best_model_path = results_dir / "fno3d_best_checkpoint.pt"

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
                model,
                train_loader,
                optimizer,
                criterion_rel,
                criterion_mse,
                device,
                progress,
                task_id,
                train_on_residual_derivative=train_on_residual_derivative,
            )
            val_loss_rel, val_loss_mse, val_diagnostics = validate(
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
            sampled_pair_diagnostics = validate_sampled_pairs(
                model,
                test_loader.dataset,
                criterion_rel,
                criterion_mse,
                device,
                batch_size=test_loader.batch_size or 1,
            )
            val_diagnostics["sampled_pairs"] = sampled_pair_diagnostics
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch + 1,
            "train/rel_l2": train_loss_rel,
            "train/mse": train_loss_mse,
            "train/on_residual_derivative": float(train_on_residual_derivative),
            "test/rel_l2": val_loss_rel,
            "test/mse": val_loss_mse,
            "test/identity_rel_l2": val_diagnostics["identity_rel_l2"],
            "test/identity_mse": val_diagnostics["identity_mse"],
            "test/model_identity_ratio": val_diagnostics["model_identity_ratio"],
            "test_sampled/rel_l2": sampled_pair_diagnostics["rel_l2"],
            "test_sampled/mse": sampled_pair_diagnostics["mse"],
            "test_sampled/identity_rel_l2": sampled_pair_diagnostics["identity_rel_l2"],
            "test_sampled/identity_mse": sampled_pair_diagnostics["identity_mse"],
            "test_sampled/model_identity_ratio": sampled_pair_diagnostics["model_identity_ratio"],
            "test_sampled/num_samples": sampled_pair_diagnostics["num_samples"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": time.time() - t_epoch_start,
        }
        for bin_name, bin_metrics in val_diagnostics["delta_a_bins"].items():
            if bin_metrics["count"] > 0:
                epoch_metrics[f"test/delta_a_{bin_name}/rel_l2"] = bin_metrics["model_rel_l2"]
                epoch_metrics[f"test/delta_a_{bin_name}/identity_rel_l2"] = bin_metrics["identity_rel_l2"]
                epoch_metrics[f"test/delta_a_{bin_name}/model_identity_ratio"] = bin_metrics["model_identity_ratio"]
                epoch_metrics[f"test/delta_a_{bin_name}/count"] = bin_metrics["count"]
        for bin_name, bin_metrics in sampled_pair_diagnostics["delta_a_bins"].items():
            if bin_metrics["count"] > 0:
                epoch_metrics[f"test_sampled/delta_a_{bin_name}/rel_l2"] = bin_metrics["model_rel_l2"]
                epoch_metrics[f"test_sampled/delta_a_{bin_name}/identity_rel_l2"] = bin_metrics["identity_rel_l2"]
                epoch_metrics[f"test_sampled/delta_a_{bin_name}/model_identity_ratio"] = bin_metrics["model_identity_ratio"]
                epoch_metrics[f"test_sampled/delta_a_{bin_name}/count"] = bin_metrics["count"]

        current_ratio = sampled_pair_diagnostics["model_identity_ratio"]
        is_best = current_ratio < best_ratio
        if is_best:
            best_ratio = current_ratio
            best_epoch = epoch + 1
            best_val_loss_rel = val_loss_rel
            best_val_loss_mse = val_loss_mse
            best_val_diagnostics = copy.deepcopy(val_diagnostics)
            torch.save(model.state_dict(), best_model_path)

        epoch_metrics["best/epoch"] = best_epoch
        epoch_metrics["best/test_sampled_model_identity_ratio"] = best_ratio
        epoch_metrics["best/is_current_epoch"] = float(is_best)
        if run is not None:
            run.log(epoch_metrics, step=epoch + 1)

        console.print(
            f"Epoch {epoch + 1:03d}/{epochs:03d} | "
            f"train rel_l2={train_loss_rel:.4f} mse={train_loss_mse:.6f} | "
            f"test rel_l2={val_loss_rel:.4f} mse={val_loss_mse:.6f} | "
            f"identity rel_l2={val_diagnostics['identity_rel_l2']:.4f} "
            f"ratio={val_diagnostics['model_identity_ratio']:.3f} | "
            f"sampled rel_l2={sampled_pair_diagnostics['rel_l2']:.4f} "
            f"identity={sampled_pair_diagnostics['identity_rel_l2']:.4f} "
            f"ratio={sampled_pair_diagnostics['model_identity_ratio']:.3f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"time={epoch_metrics['epoch_time_sec']:.1f}s"
        )

    model_path = results_dir / "fno3d_checkpoint.pt"
    torch.save(model.state_dict(), model_path)
    log_artifact(model_path, "model")
    print(f"\nSaved model checkpoint to: {model_path}")
    if best_model_path.exists():
        try:
            best_state = torch.load(best_model_path, map_location=device, weights_only=True)
        except TypeError:
            best_state = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_state)
        log_artifact(best_model_path, "model")
        print(
            f"Saved best model checkpoint to: {best_model_path} "
            f"(epoch {best_epoch}, test_sampled ratio={best_ratio:.3f})"
        )
        if best_val_diagnostics is not None:
            best_val_diagnostics["best_checkpoint"] = {
                "path": str(best_model_path),
                "epoch": best_epoch,
                "test_sampled_model_identity_ratio": best_ratio,
            }
        return best_val_loss_rel, best_val_loss_mse, best_val_diagnostics

    return val_loss_rel, val_loss_mse, val_diagnostics


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
    gabriela_identity_errors = []
    gabriela_pair_metrics = []
    bin_stats = _empty_bin_stats()
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

            pred64 = predict_residual(model, x_in).cpu().squeeze(0).squeeze(0)

            pred256 = F.interpolate(pred64.unsqueeze(0).unsqueeze(0), size=(256, 256, 256), mode="trilinear", align_corners=True).squeeze(0).squeeze(0)
            identity256 = F.interpolate(rho64_in.unsqueeze(0).unsqueeze(0), size=(256, 256, 256), mode="trilinear", align_corners=True).squeeze(0).squeeze(0)

            diff_norm = torch.linalg.vector_norm(pred256 - rho256_out, ord=2)
            identity_diff_norm = torch.linalg.vector_norm(identity256 - rho256_out, ord=2)
            target_norm = torch.linalg.vector_norm(rho256_out, ord=2)
            rel_error = (diff_norm / (target_norm + 1e-8)).item()
            identity_rel_error = (identity_diff_norm / (target_norm + 1e-8)).item()
            ratio = rel_error / (identity_rel_error + 1e-12)
            gabriela_errors.append(rel_error)
            gabriela_identity_errors.append(identity_rel_error)

            bin_name = _delta_a_bin(delta_a)
            stats = bin_stats[bin_name]
            stats["count"] += 1
            stats["model_rel_sum"] += rel_error
            stats["identity_rel_sum"] += identity_rel_error

            gabriela_pair_metrics.append({
                "pair_index": pair_idx + 1,
                "z0": z0,
                "z1": z1,
                "delta_a": delta_a,
                "delta_a_bin": bin_name,
                "model_rel_l2": rel_error,
                "identity_rel_l2": identity_rel_error,
                "model_identity_ratio": ratio,
            })

            print(
                f"  Pair {pair_idx+1:02d}/{len(g_pairs)}: z={z0:.2f}->{z1:.2f} "
                f"(delta_a={delta_a:.4f}) | model={rel_error:.4f} "
                f"identity={identity_rel_error:.4f} ratio={ratio:.3f}"
            )

            if plot_pair and z0 == plot_pair[0] and z1 == plot_pair[1]:
                plot_data = {
                    "in_slice": rho256_in[:, :, 128].numpy(),
                    "target_slice": rho256_out[:, :, 128].numpy(),
                    "pred_slice": pred256[:, :, 128].numpy(),
                    "z0": z0,
                    "z1": z1
                }

    mean_g_error = np.mean(gabriela_errors)
    mean_g_identity_error = np.mean(gabriela_identity_errors)
    mean_g_ratio = mean_g_error / (mean_g_identity_error + 1e-12)
    gabriela_bin_metrics = _finalize_bin_stats(bin_stats)
    print(f"\nMean Relative L2 Error on Gabriela Dataset: {mean_g_error:.4f}")
    print(f"Mean Identity Relative L2 Error on Gabriela Dataset: {mean_g_identity_error:.4f} | ratio={mean_g_ratio:.3f}")
    for bin_name, metrics in gabriela_bin_metrics.items():
        if metrics["count"] > 0:
            print(
                f"  Delta a {bin_name}: count={metrics['count']} "
                f"model={metrics['model_rel_l2']:.4f} "
                f"identity={metrics['identity_rel_l2']:.4f} "
                f"ratio={metrics['model_identity_ratio']:.3f}"
            )
    diagnostics = {
        "identity_errors": gabriela_identity_errors,
        "mean_identity_rel_l2": mean_g_identity_error,
        "mean_model_identity_ratio": mean_g_ratio,
        "delta_a_bins": gabriela_bin_metrics,
        "pair_metrics": gabriela_pair_metrics,
    }
    if wandb.run is not None:
        wandb_metrics = {
            "gabriela/mean_rel_l2": mean_g_error,
            "gabriela/mean_identity_rel_l2": mean_g_identity_error,
            "gabriela/mean_model_identity_ratio": mean_g_ratio,
        }
        for bin_name, metrics in gabriela_bin_metrics.items():
            if metrics["count"] > 0:
                wandb_metrics[f"gabriela/delta_a_{bin_name}/rel_l2"] = metrics["model_rel_l2"]
                wandb_metrics[f"gabriela/delta_a_{bin_name}/identity_rel_l2"] = metrics["identity_rel_l2"]
                wandb_metrics[f"gabriela/delta_a_{bin_name}/model_identity_ratio"] = metrics["model_identity_ratio"]
                wandb_metrics[f"gabriela/delta_a_{bin_name}/count"] = metrics["count"]
        wandb.log(wandb_metrics)
    return gabriela_errors, plot_data, mean_g_error, diagnostics


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
        pred_sample = predict_residual(model, x_in).cpu().squeeze(0).squeeze(0)

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


def save_metrics(val_loss_rel, val_loss_mse, val_diagnostics, gabriela_errors, mean_g_error, gabriela_diagnostics, results_dir):
    metrics = {
        "miguel_64_test_val_rel_l2": val_loss_rel,
        "miguel_64_test_val_mse": val_loss_mse,
        "miguel_64_test_identity_rel_l2": val_diagnostics["identity_rel_l2"],
        "miguel_64_test_identity_mse": val_diagnostics["identity_mse"],
        "miguel_64_test_model_identity_ratio": val_diagnostics["model_identity_ratio"],
        "miguel_64_test_delta_a_bins": val_diagnostics["delta_a_bins"],
        "miguel_64_test_sampled_pairs": val_diagnostics.get("sampled_pairs"),
        "best_checkpoint": val_diagnostics.get("best_checkpoint"),
        "gabriela_test_mean_rel_l2": mean_g_error,
        "gabriela_test_mean_identity_rel_l2": gabriela_diagnostics["mean_identity_rel_l2"],
        "gabriela_test_mean_model_identity_ratio": gabriela_diagnostics["mean_model_identity_ratio"],
        "gabriela_test_delta_a_bins": gabriela_diagnostics["delta_a_bins"],
        "gabriela_pair_errors": gabriela_errors,
        "gabriela_pair_identity_errors": gabriela_diagnostics["identity_errors"],
        "gabriela_pair_metrics": gabriela_diagnostics["pair_metrics"],
    }
    with open(results_dir / "fno3d_evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Saved evaluation metrics JSON.")
