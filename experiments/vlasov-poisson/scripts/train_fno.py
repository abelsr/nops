import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from datetime import datetime
from torch.utils.data import DataLoader
import mlflow

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
from vlasov_poisson.utils.mlflow_utils import setup_mlflow


def main():
    args = get_args()
    config = load_config(args.config)
    training_config = config.get("training", {})

    batch_size = int(get_training_value(config, args, "batch_size", default=4))
    epochs = int(get_training_value(config, args, "num_epochs", arg_name="epochs", default=15))
    epochs = 2 if args.dry_run else epochs
    lr = float(get_training_value(config, args, "learning_rate", arg_name="lr", default=1e-3))
    max_step = int(get_training_value(config, args, "max_step", default=50))
    weight_decay = float(training_config.get("weight_decay", 1e-4))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Config: {args.config}")
    print(f"Device: {device}")

    data_dir, results_dir = get_paths()

    setup_mlflow(tracking_dir=args.mlruns_dir)

    print("Loading Miguel_64 dataset...")
    m_train_data, m_train_as, m_test_data, m_test_as = load_miguel_data(data_dir, args.dry_run, args.lazy_load)

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

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("config", args.config)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", lr)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("max_step", max_step)

        val_loss_rel, val_loss_mse = train(
            model, train_loader, test_loader, optimizer, scheduler,
            criterion_rel, criterion_mse, epochs, device, results_dir, run_name=run_name
        )

        gabriela_errors, plot_data, mean_g_error = evaluate_gabriela(model, data_dir, device)

        plot_gabriela_comparison(plot_data, results_dir)
        plot_miguel_comparison(model, test_dataset, device, results_dir)

        save_metrics(val_loss_rel, val_loss_mse, gabriela_errors, mean_g_error, results_dir)

        gabriela_plot = results_dir / "gabriela_fno_prediction.png"
        if gabriela_plot.exists():
            mlflow.log_artifact(str(gabriela_plot))

        miguel_plot = results_dir / "miguel_fno_prediction.png"
        if miguel_plot.exists():
            mlflow.log_artifact(str(miguel_plot))

        metrics_file = results_dir / "fno3d_evaluation_metrics.json"
        if metrics_file.exists():
            mlflow.log_artifact(str(metrics_file))


if __name__ == "__main__":
    main()
