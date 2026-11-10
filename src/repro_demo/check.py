"""Compare this code's result with the golden value, in plain words.

    python -m repro_demo.check

Does the same comparison as tests/golden, but prints the numbers side by side.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from repro_demo.train import TrainConfig, train

GOLDEN = Path("tests/golden/golden.json")


def main() -> int:
    golden = json.loads(GOLDEN.read_text())
    acc = train(TrainConfig(seed=42))["val_accuracy"]
    diff = acc - golden["val_accuracy"]
    ok = abs(diff) <= golden["tolerance"]
    print(f"golden accuracy (committed) : {golden['val_accuracy']:.4f}")
    print(f"accuracy of this code       : {acc:.4f}")
    print(f"difference                  : {diff:+.4f}   (allowed: ±{golden['tolerance']})")
    if ok:
        print("PASS: the result matches the golden value")
    else:
        print("FAIL: the result moved. Was this change intended?")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
