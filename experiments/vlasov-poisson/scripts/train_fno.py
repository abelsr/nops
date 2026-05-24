import torch
from datetime import datetime
from torch.utils.data import DataLoader

from .config import get_args, get_paths
from .data import VlasovPoissonDataset
from .losses import RelativeL2Loss
from .trainer import (
    load_miguel_data,
    create_model,
    train,
    evaluate_gabriela,
    plot_gabriela_comparison,
    plot_miguel_comparison,
    save_metrics,
)
from .mlflow_utils import setup_mlflow


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir, results_dir = get_paths()

    setup_mlflow()

    print("Loading Miguel_64 dataset...")
    m_train_data, m_train_as, m_test_data, m_test_as = load_miguel_data(data_dir, args.dry_run)

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

    print("\nInstantiating FNO3D model...")
    model = create_model(device)

    criterion_rel = RelativeL2Loss()
    criterion_mse = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    epochs = 2 if args.dry_run else args.epochs
    print(f"\nStarting training for {epochs} epochs...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"fno3d-vlasov-{timestamp}"

    val_loss_rel, val_loss_mse = train(
        model, train_loader, test_loader, optimizer, scheduler,
        criterion_rel, criterion_mse, epochs, device, results_dir, run_name=run_name
    )

    mlflow.log_param("batch_size", args.batch_size)
    mlflow.log_param("lr", args.lr)
    mlflow.log_param("max_step", args.max_step)

    gabriela_errors, plot_data, mean_g_error = evaluate_gabriela(model, data_dir, device)

    plot_gabriela_comparison(plot_data, results_dir)
    plot_miguel_comparison(model, test_dataset, device, results_dir)

    save_metrics(val_loss_rel, val_loss_mse, gabriela_errors, mean_g_error, results_dir)


if __name__ == "__main__":
    main()