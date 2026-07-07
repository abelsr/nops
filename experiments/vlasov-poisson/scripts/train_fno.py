import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from datetime import datetime
from torch.utils.data import DataLoader
import wandb

from vlasov_poisson.config.schema import get_args, get_paths, get_training_value, load_config
from vlasov_poisson.datasets.dataset import VlasovPoissonDataset, load_miguel_data
from vlasov_poisson.losses.spectral import RelativeL2Loss
from vlasov_poisson.models.fno import FNOConfig, create_model
from vlasov_poisson.trainers.trainer import (
    train,
    evaluate_gabriela,
    plot_gabriela_comparison,
    plot_miguel_comparison,
    save_metrics,
)
from vlasov_poisson.utils.wandb_utils import init_wandb, log_artifact


def main():
    args = get_args()
    config = load_config(args.config)
    training_config = config.get("training", {})
    wandb_config_file = config.get("logging", {}).get("wandb", {})

    batch_size = int(get_training_value(config, args, "batch_size", default=4))
    epochs = int(get_training_value(config, args, "num_epochs", arg_name="epochs", default=15))
    epochs = 2 if args.dry_run else epochs
    lr = float(get_training_value(config, args, "learning_rate", arg_name="lr", default=1e-3))
    max_step = int(get_training_value(config, args, "max_step", default=50))
    weight_decay = float(training_config.get("weight_decay", 1e-4))
    split_strategy = training_config.get("split_strategy", "chronological")
    train_fraction = float(training_config.get("train_fraction", 0.8))
    train_on_residual_derivative = bool(training_config.get("train_on_residual_derivative", False))
    wandb_project = args.wandb_project or wandb_config_file.get("project", "vlasov-poisson")
    wandb_entity = args.wandb_entity or wandb_config_file.get("entity")
    wandb_mode = args.wandb_mode or wandb_config_file.get("mode", "online")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Config: {args.config}")
    print(f"Device: {device}")

    data_dir, results_dir = get_paths()

    print("Loading Miguel_64 dataset...")
    m_train_data, m_train_as, m_test_data, m_test_as = load_miguel_data(
        data_dir,
        args.dry_run,
        args.lazy_load,
        split_strategy=split_strategy,
        train_fraction=train_fraction,
    )

    train_dataset = VlasovPoissonDataset(
        m_train_data, m_train_as, is_train=True,
        max_step=max_step, num_samples=1000 if args.dry_run else int(training_config.get("num_samples", 5000)),
        lazy=args.lazy_load
    )
    test_dataset = VlasovPoissonDataset(
        m_test_data, m_test_as, is_train=False,
        max_step=max_step, lazy=args.lazy_load
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("\nInstantiating FNO3D model...")
    model_config = FNOConfig.from_dict(config.get("model", {}).get("params", {}))
    model = create_model(model_config, device)

    criterion_rel = RelativeL2Loss()
    criterion_mse = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler_config = training_config.get("scheduler", {})
    scheduler_params = dict(scheduler_config.get("params", {}))
    scheduler_params.setdefault("T_max", epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)

    print(f"\nStarting training for {epochs} epochs...")
    print(f"Batch size: {batch_size} | lr: {lr:g} | weight decay: {weight_decay:g} | max_step: {max_step}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"fno3d-vlasov-{timestamp}"

    wandb_config = {
        **config,
        "training": {
            **training_config,
            "batch_size": batch_size,
            "num_epochs": epochs,
            "learning_rate": lr,
            "max_step": max_step,
            "weight_decay": weight_decay,
            "split_strategy": split_strategy,
            "train_fraction": train_fraction,
            "train_on_residual_derivative": train_on_residual_derivative,
            "dry_run": args.dry_run,
            "lazy_load": args.lazy_load,
        },
        "device": str(device),
    }

    with init_wandb(
        project=wandb_project,
        entity=wandb_entity,
        mode=wandb_mode,
        run_name=run_name,
        config=wandb_config,
        config_path=args.config,
    ) as run:

        val_loss_rel, val_loss_mse, val_diagnostics = train(
            model, train_loader, test_loader, optimizer, scheduler,
            criterion_rel, criterion_mse, epochs, device, results_dir, run=run,
            train_on_residual_derivative=train_on_residual_derivative,
        )

        gabriela_errors, plot_data, mean_g_error, gabriela_diagnostics = evaluate_gabriela(model, data_dir, device)

        plot_gabriela_comparison(plot_data, results_dir)
        plot_miguel_comparison(model, test_dataset, device, results_dir)

        save_metrics(val_loss_rel, val_loss_mse, val_diagnostics, gabriela_errors, mean_g_error, gabriela_diagnostics, results_dir)

        gabriela_plot = results_dir / "gabriela_fno_prediction.png"
        if gabriela_plot.exists():
            wandb.log({"gabriela_prediction": wandb.Image(str(gabriela_plot))})
            log_artifact(gabriela_plot, "plot")

        miguel_plot = results_dir / "miguel_fno_prediction.png"
        if miguel_plot.exists():
            wandb.log({"miguel_prediction": wandb.Image(str(miguel_plot))})
            log_artifact(miguel_plot, "plot")

        metrics_file = results_dir / "fno3d_evaluation_metrics.json"
        if metrics_file.exists():
            log_artifact(metrics_file, "metrics")


if __name__ == "__main__":
    main()
