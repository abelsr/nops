import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="Train FNO3D on Vlasov-Poisson cosmological simulations.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-step", type=int, default=50, help="Maximum snapshot index difference for a pair")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick dry run to verify the training pipeline")
    parser.add_argument("--mlruns-dir", type=str, default="./mlruns", help="MLflow tracking directory")
    return parser.parse_args()


def get_paths():
    base_dir = Path("experiments/vlasov-poisson")
    data_dir = base_dir / "data" / "simulations"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    return data_dir, results_dir