import argparse
from pathlib import Path

import yaml


def get_args():
    parser = argparse.ArgumentParser(description="Train FNO3D on Vlasov-Poisson cosmological simulations.")
    parser.add_argument("--config", type=str, default="experiments/vlasov-poisson/configs/v1.yaml", help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--max-step", type=int, default=None, help="Maximum snapshot index difference for a pair")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick dry run to verify the training pipeline")
    parser.add_argument("--lazy-load", action="store_true", help="Load data on-demand instead of pre-loading all snapshots")
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/team name")
    parser.add_argument("--wandb-mode", type=str, default=None, choices=["online", "offline", "disabled"], help="W&B logging mode")
    return parser.parse_args()


def load_config(config_path):
    path = Path(config_path)
    with path.open("r") as f:
        config = yaml.safe_load(f) or {}
    return config


def get_training_value(config, args, key, arg_name=None, default=None):
    arg_name = arg_name or key
    cli_value = getattr(args, arg_name)
    if cli_value is not None:
        return cli_value
    return config.get("training", {}).get(key, default)


def get_paths():
    base_dir = Path("experiments/vlasov-poisson")
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    return data_dir, results_dir
