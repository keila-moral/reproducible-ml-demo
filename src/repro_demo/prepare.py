"""Preprocessing as code: build the training table from a fixed source.

Run through DVC (`dvc repro`) so the output's content hash is recorded in dvc.lock.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_digits


def build_table() -> pd.DataFrame:
    """8x8 handwritten digits (1,797 images), shipped with scikit-learn: no download needed."""
    digits = load_digits()
    df = pd.DataFrame(digits.data, columns=[f"p{i}" for i in range(digits.data.shape[1])])
    df["label"] = digits.target
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/digits.parquet")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build_table().to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
