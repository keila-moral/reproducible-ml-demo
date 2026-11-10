"""Train once in a fresh process and log the run to MLflow.

    python -m repro_demo.run --seed 42 --name seeded-a
    python -m repro_demo.run --no-seed --name unseeded

Each call is a new Python process, like a rerun on another day: nothing is
carried over from earlier runs in a notebook session.
"""
from __future__ import annotations

import argparse

import mlflow

from repro_demo.lineage import log_run
from repro_demo.train import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-seed", action="store_true")
    parser.add_argument("--name", default=None)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="devfest-reproducibility")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    cfg = TrainConfig(seed=None if args.no_seed else args.seed)
    metrics = train(cfg)
    log_run(cfg, metrics, run_name=args.name)
    print(args.name or "run", metrics)


if __name__ == "__main__":
    main()
