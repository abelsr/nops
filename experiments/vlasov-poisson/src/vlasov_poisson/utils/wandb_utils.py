from pathlib import Path

import wandb


def init_wandb(project, entity, mode, run_name, config, config_path):
    return wandb.init(
        project=project,
        entity=entity,
        mode=mode,
        name=run_name,
        config={**config, "config_path": config_path},
    )


def log_artifact(path, artifact_type):
    path = Path(path)
    if not path.exists() or wandb.run is None:
        return

    artifact = wandb.Artifact(name=path.stem, type=artifact_type)
    artifact.add_file(str(path))
    wandb.log_artifact(artifact)
