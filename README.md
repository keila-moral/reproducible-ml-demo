# It Worked in My Notebook

[![golden-result](https://github.com/keila-moral/reproducible-ml-demo/actions/workflows/golden.yml/badge.svg)](https://github.com/keila-moral/reproducible-ml-demo/actions/workflows/golden.yml)

Companion code for the talk **"It Worked in My Notebook: Engineering Reproducible ML Pipelines"**
by Keila Moral Figueroa, DevFest Berlin 2026.

An ML result can change without a single line of your code changing. This repository shows four
places where that drift hides, and one engineering control for each, on a model small enough
to train on a CPU in about a minute: a Keras model (JAX backend) trained on scikit-learn's built-in
8×8 digits dataset.

| Drift | Control | Tool |
| --- | --- | --- |
| Dependencies change under you | Lock exact versions and file hashes | [uv](https://docs.astral.sh/uv/) |
| Data and preprocessing change | Version the data, make preprocessing a pipeline stage | [DVC](https://dvc.org/) |
| Randomness and numerics differ | Make seeds an input, record the hardware | JAX PRNG keys, [MLflow](https://mlflow.org/) |
| Hosted models change | Pin the model version, log it, cache responses | [Gemini API](https://ai.google.dev/) |

On top of all four, a **golden-result test** in CI fails a pull request when the result moves,
even if nothing crashed.

## The notebooks

Each notebook runs on its own in Google Colab. Open one, run the cells from the top, and follow the
instructions in its first cell.

| | Notebook | What it shows |
| --- | --- | --- |
| 00 | [The silent failure](https://colab.research.google.com/github/keila-moral/reproducible-ml-demo/blob/main/notebooks/00_the_silent_failure.ipynb) | Same code, three runs, three accuracies; then a seed makes them agree |
| 01 | [A locked environment](https://colab.research.google.com/github/keila-moral/reproducible-ml-demo/blob/main/notebooks/01_locked_environment.ipynb) | Replacing Colab's preinstalled packages with the hash-verified `uv.lock`, and checking it |
| 02 | [Two runs, one number](https://colab.research.google.com/github/keila-moral/reproducible-ml-demo/blob/main/notebooks/02_lineage_and_determinism.ipynb) | DVC data hashes, JAX PRNG keys, and identical seeded runs recorded in MLflow |
| 03 | [Pinned versus newest](https://colab.research.google.com/github/keila-moral/reproducible-ml-demo/blob/main/notebooks/03_gemini_pinned_vs_latest.ipynb) | The same evaluation on a pinned Gemini model and on "whatever is newest", logged and cached |
| 04 | [The red pull request](https://colab.research.google.com/github/keila-moral/reproducible-ml-demo/blob/main/notebooks/04_golden_result_test.ipynb) | A golden-result test catching a harmless-looking refactor |

Notebook 03 needs a Gemini API key, stored as a Colab secret named `GOOGLE_API_KEY`
(key icon in the left sidebar). The free tier is enough. Never paste the key into a cell.

## Run it locally

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.11 or newer.

```bash
git clone https://github.com/keila-moral/reproducible-ml-demo.git
cd reproducible-ml-demo
uv sync --locked                 # install exactly what uv.lock records
uv run dvc repro                 # rebuild the dataset; dvc.lock holds its hash
uv run pytest tests/golden -v    # train once and compare with the golden value
```

A few command-line entry points do the same steps as the notebooks:

```bash
uv run python -m repro_demo.run --seed 42 --name seeded-a   # one run, logged to MLflow
uv run python -m repro_demo.run --no-seed --name unseeded   # the silent failure
uv run python -m repro_demo.check                           # golden comparison, in plain words
uv run python -m repro_demo.make_golden                     # regenerate the golden value
```

## The golden-result test

[`tests/golden/test_result.py`](tests/golden/test_result.py) trains with `seed=42` and checks that
validation accuracy is within a tolerance of the committed value in
[`tests/golden/golden.json`](tests/golden/golden.json) (currently 0.958217 ± 0.005).

[`.github/workflows/golden.yml`](.github/workflows/golden.yml) runs it on every pull request, and
a branch rule makes it a required check before merging into `main`. It also checks that the rebuilt
data matches the hash in `dvc.lock`. Dependabot opens weekly dependency-update pull requests, so
each upgrade is tested against the result, not only against whether the code runs.

If a change moves the result on purpose, run `make_golden` and commit the new `golden.json`.
The diff then shows reviewers, in plain numbers, how the accuracy changed.

## Repository layout

```
src/repro_demo/
  prepare.py       build the training table (the DVC stage)
  train.py         TrainConfig and train(), shared by every notebook and the test
  lineage.py       record commit, environment, data hash and config for a run
  run.py           one training run in a fresh process, logged to MLflow
  check.py         compare this code's result with the golden value
  make_golden.py   regenerate tests/golden/golden.json
  timing.py        measure the cost of deterministic GPU kernels
tests/golden/      the golden-result test and its committed value
notebooks/         the five Colab notebooks
dvc.yaml, dvc.lock the data pipeline and the recorded hash of its output
uv.lock            every package, its exact version and file hashes
```

## License

[MIT](LICENSE). The digits dataset ships with scikit-learn (BSD-3-Clause).
