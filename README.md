# reproducible-ml-demo

Demo code for the DevFest Berlin 2026 talk
**"It Worked in My Notebook: Engineering Reproducible ML Pipelines"** (Keila Moral Figueroa).

A small Keras model (JAX backend) on scikit-learn's built-in digits dataset, used to show four
sources of silent drift and one control for each.

| Notebook | Shows |
| --- | --- |
| `notebooks/00_the_silent_failure.ipynb` | Same code, different accuracy on every unseeded run |
| `notebooks/01_locked_environment.ipynb` | Installing the hash-verified `uv.lock` in Colab and checking it |
| `notebooks/02_lineage_and_determinism.ipynb` | DVC data hashes, JAX PRNG keys, identical seeded runs in MLflow |
| `notebooks/03_gemini_pinned_vs_latest.ipynb` | Pinned Gemini model vs `-latest` alias, logged and cached |
| `notebooks/04_golden_result_test.ipynb` | A golden-result test catching a harmless-looking refactor |

## Run locally

```bash
uv sync --locked
uv run dvc repro
uv run pytest tests/golden -v
```

`.github/workflows/golden.yml` runs the same test on every pull request that touches the lockfile,
the source or the data definition.
