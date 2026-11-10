"""Record what produced a result: code, environment, data and config."""
from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path

import mlflow

from repro_demo.train import TrainConfig


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def file_md5(path: str | Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def environment() -> dict:
    import jax
    import keras

    return {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "keras": keras.__version__,
        "keras_backend": keras.backend.backend(),
        "devices": ",".join(str(d) for d in jax.devices()),
    }


def log_run(cfg: TrainConfig, metrics: dict, run_name: str | None = None) -> str:
    """Log one training run to MLflow and return its run id."""
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("git_commit", git_commit())
        mlflow.set_tags({f"env.{k}": v for k, v in environment().items()})
        mlflow.log_params(cfg.as_params())
        mlflow.log_param("data_md5", file_md5(cfg.data_path))
        for lock in ("uv.lock", "dvc.lock"):
            if Path(lock).exists():
                mlflow.log_artifact(lock)
        mlflow.log_metrics(metrics)
        return run.info.run_id
