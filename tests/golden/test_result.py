"""Golden-result test: fails when the *number* changes, not only when the code crashes."""
import json
from pathlib import Path

from repro_demo.train import TrainConfig, train

GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text())


def test_golden_result():
    acc = train(TrainConfig(seed=42))["val_accuracy"]
    assert abs(acc - GOLDEN["val_accuracy"]) <= GOLDEN["tolerance"], (
        f"val_accuracy {acc:.4f} drifted from golden {GOLDEN['val_accuracy']:.4f} "
        f"(tolerance {GOLDEN['tolerance']})"
    )
