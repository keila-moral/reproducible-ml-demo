"""Measure the cost of deterministic GPU kernels.

XLA_FLAGS is read when JAX initialises, so each setting needs a fresh process:

    python -m repro_demo.timing                     # default kernels
    XLA_FLAGS=--xla_gpu_deterministic_ops=true python -m repro_demo.timing
"""
from __future__ import annotations

import os
import time

from repro_demo.train import TrainConfig, train


def main(repeats: int = 3) -> None:
    import jax

    cfg = TrainConfig(seed=42, epochs=30)
    train(cfg)  # warm-up: excludes compilation time
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        metrics = train(cfg)
        times.append(time.perf_counter() - start)
    print(f"devices={jax.devices()} XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r}")
    print(f"best of {repeats}: {min(times):.2f} s  metrics={metrics}")


if __name__ == "__main__":
    main()
