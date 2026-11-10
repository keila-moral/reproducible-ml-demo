"""Regenerate tests/golden/golden.json. Run in the CI environment, then commit the result."""
from __future__ import annotations

import json
from pathlib import Path

from repro_demo.train import TrainConfig, train

GOLDEN = Path("tests/golden/golden.json")

if __name__ == "__main__":
    metrics = train(TrainConfig(seed=42))
    GOLDEN.write_text(json.dumps({"val_accuracy": metrics["val_accuracy"], "tolerance": 0.005}, indent=2) + "\n")
    print(GOLDEN.read_text())
