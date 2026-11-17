"""Golden-result tests: compare one seeded training run with committed values.

Two separate questions, two separate assertions:

* ``test_accuracy_within_tolerance`` asks "same score?". It is a tolerance-based
  regression check on one metric. With 359 validation samples, one prediction is
  worth about 0.0028 accuracy, so a +-0.005 tolerance lets one changed prediction
  (sometimes more, if errors cancel) pass.
* ``test_predictions_identical`` asks "same behaviour?". It compares a sha256 of
  the predicted label for every validation sample, so it fails if any single
  prediction changes, even when the accuracy is identical.

Neither test detects data distribution drift: both run on the fixed, versioned
validation set. A green run means these two assertions held for this code, this
environment and this data, and nothing more.
"""
import json
from pathlib import Path

import pytest

from repro_demo.train import TrainConfig, predictions_sha256, train_and_predict

GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text())


@pytest.fixture(scope="module")
def run():
    """Train once and share the result between both tests."""
    metrics, predictions = train_and_predict(TrainConfig(seed=42))
    return metrics, predictions


def test_accuracy_within_tolerance(run):
    acc = run[0]["val_accuracy"]
    assert abs(acc - GOLDEN["val_accuracy"]) <= GOLDEN["tolerance"], (
        f"val_accuracy {acc:.4f} drifted from golden {GOLDEN['val_accuracy']:.4f} "
        f"(tolerance {GOLDEN['tolerance']})"
    )


def test_predictions_identical(run):
    predictions = run[1]
    assert predictions.size == GOLDEN["n_val"], (
        f"{predictions.size} validation predictions, golden has {GOLDEN['n_val']}"
    )
    digest = predictions_sha256(predictions)
    assert digest == GOLDEN["val_predictions_sha256"], (
        f"validation predictions changed: sha256 {digest[:12]}... "
        f"vs golden {GOLDEN['val_predictions_sha256'][:12]}..."
    )
