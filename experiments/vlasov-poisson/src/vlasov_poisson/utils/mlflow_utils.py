from pathlib import Path

import mlflow


def setup_mlflow(tracking_dir=None):
    if tracking_dir:
        resolved = str(Path(tracking_dir).resolve())
        mlflow.set_tracking_uri(resolved)
    mlflow.set_experiment("FNO-3D Vlasov-Poisson")
