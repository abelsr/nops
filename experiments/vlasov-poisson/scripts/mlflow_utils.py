import mlflow
import torch
from pathlib import Path


def setup_mlflow(experiment_name="vlasov-fno-training", tracking_dir="./mlruns"):
    mlflow.set_tracking_uri(f"file://{tracking_dir}")
    mlflow.set_experiment(experiment_name)


def get_next_model_version() -> int:
    client = mlflow.MlflowClient()
    try:
        versions = client.get_model_version_by_name("fno3d-vlasov", "latest")
        return int(versions.version) + 1
    except Exception:
        return 1


def log_run(model, params, metrics, artifacts_dir=None):
    with mlflow.start_run() as run:
        mlflow.log_params(params)

        for key, value in metrics.items():
            if isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    mlflow.log_metric(key, v, step=i)
            else:
                mlflow.log_metric(key, value)

        if artifacts_dir:
            artifacts_path = Path(artifacts_dir)
            for f in artifacts_path.glob("*"):
                mlflow.log_artifact(str(f))

        if model is not None:
            mlflow.pytorch.log_model(
                model,
                "fno3d-model",
                registered_model_name="fno3d-vlasov"
            )

    return run.info.run_id