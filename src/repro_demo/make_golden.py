"""Regenerate tests/golden/golden.json. Run in the CI environment, then commit the result."""
from __future__ import annotations

import json
from pathlib import Path

from repro_demo.train import TrainConfig, predictions_sha256, train_and_predict

GOLDEN = Path("tests/golden/golden.json")

if __name__ == "__main__":
    metrics, predictions = train_and_predict(TrainConfig(seed=42))
    golden = {
        "val_accuracy": metrics["val_accuracy"],
        "tolerance": 0.005,
        "n_val": int(predictions.size),
        "val_predictions_sha256": predictions_sha256(predictions),
    }
    GOLDEN.write_text(json.dumps(golden, indent=2) + "\n")
    print(GOLDEN.read_text())
