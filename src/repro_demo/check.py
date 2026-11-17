"""Compare this code's result with the golden values, in plain words.

    python -m repro_demo.check

Does the same comparisons as tests/golden, but prints them side by side:
"same score?" (accuracy within tolerance) and "same behaviour?" (every
validation prediction identical, compared through a sha256 of the labels).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from repro_demo.train import TrainConfig, predictions_sha256, train_and_predict

GOLDEN = Path("tests/golden/golden.json")


def main() -> int:
    golden = json.loads(GOLDEN.read_text())
    metrics, predictions = train_and_predict(TrainConfig(seed=42))
    acc = metrics["val_accuracy"]
    diff = acc - golden["val_accuracy"]
    same_score = abs(diff) <= golden["tolerance"]
    digest = predictions_sha256(predictions)
    same_behaviour = digest == golden["val_predictions_sha256"]

    print(f"golden accuracy (committed) : {golden['val_accuracy']:.4f}")
    print(f"accuracy of this code       : {acc:.4f}")
    print(f"difference                  : {diff:+.4f}   (allowed: ±{golden['tolerance']})")
    print(f"same score?                 : {'yes' if same_score else 'NO'}")
    print()
    print(f"golden predictions sha256   : {golden['val_predictions_sha256'][:16]}...")
    print(f"predictions of this code    : {digest[:16]}...   ({predictions.size} labels)")
    print(f"same behaviour?             : {'yes' if same_behaviour else 'NO'}")
    print()
    if same_score and same_behaviour:
        print("PASS: same score and same predictions as the golden run")
    elif same_score:
        print("FAIL: the score is within tolerance, but some predictions changed. Was this intended?")
    else:
        print("FAIL: the result moved. Was this change intended?")
    return 0 if same_score and same_behaviour else 1


if __name__ == "__main__":
    sys.exit(main())
