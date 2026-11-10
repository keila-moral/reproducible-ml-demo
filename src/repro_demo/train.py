"""A small, CPU-friendly training run shared by every demo notebook.

`train(TrainConfig(seed=None))` reproduces the silent failure: every call gives a
slightly different accuracy. `train(TrainConfig(seed=42))` is the controlled run.
"""
from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

from dataclasses import asdict, dataclass
from pathlib import Path

import keras
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrainConfig:
    seed: int | None = 42
    epochs: int = 15
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden_units: int = 64
    dropout: float = 0.2
    val_fraction: float = 0.2
    data_path: str = "data/digits.parquet"

    def as_params(self) -> dict:
        return {k: ("none" if v is None else v) for k, v in asdict(self).items()}


def preprocess(features: np.ndarray) -> np.ndarray:
    """Scale raw pixel intensities (0-16) to [0, 1]."""
    return features.astype("float32") / 16.0


def load_split(cfg: TrainConfig):
    df = pd.read_parquet(cfg.data_path)
    x = preprocess(df.drop(columns="label").to_numpy())
    y = df["label"].to_numpy().astype("int32")
    # An unseeded split is a different experiment every time.
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(df))
    n_val = int(len(df) * cfg.val_fraction)
    val, tr = order[:n_val], order[n_val:]
    return (x[tr], y[tr]), (x[val], y[val])


def build_model(cfg: TrainConfig, n_features: int, n_classes: int) -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=(n_features,)),
            keras.layers.Dense(cfg.hidden_units, activation="relu"),
            keras.layers.Dropout(cfg.dropout),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(cfg: TrainConfig = TrainConfig()) -> dict:
    """Train once and return validation metrics."""
    if cfg.seed is not None:
        # Seeds Python's `random`, NumPy and the Keras backend in one call.
        keras.utils.set_random_seed(cfg.seed)
    (x_tr, y_tr), (x_val, y_val) = load_split(cfg)
    model = build_model(cfg, x_tr.shape[1], int(y_tr.max()) + 1)
    model.fit(x_tr, y_tr, epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=0)
    loss, acc = model.evaluate(x_val, y_val, verbose=0)
    return {"val_accuracy": round(float(acc), 6), "val_loss": round(float(loss), 6)}


if __name__ == "__main__":
    print(train())
