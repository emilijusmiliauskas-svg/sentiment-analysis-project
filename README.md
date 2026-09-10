# Sentiment Analysis: A Production Pipeline Around a Deliberately Small Model

[![CI](https://github.com/emilijusmiliauskas-svg/sentiment-analysis-project/actions/workflows/ci.yml/badge.svg)](https://github.com/emilijusmiliauskas-svg/sentiment-analysis-project/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A scikit-learn sentiment classifier wrapped in the machinery that turns a script into something shippable — **automated testing, linting, containerisation, and a four-job CI/CD pipeline that publishes a Docker image on every merge to `main`.**

The model is intentionally the least interesting part. It is TF-IDF into logistic regression, trained on 20 labelled sentences. **The point of this repository is everything around it.**

---

## Why the Dataset Is Tiny on Purpose

Twenty rows cannot support a meaningful claim about sentiment classification, and this repository doesn't make one. The dataset is small so that the full pipeline — train, test, lint, build, push — runs end to end in under a minute on every commit, which is what let the CI/CD design actually get exercised rather than described.

Swapping in a real corpus is a one-line change to `--data`. The interesting engineering doesn't change.

## The Pipeline

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs four jobs:

| Job | What it does |
|-----|--------------|
| **run-predict** | Trains the model from raw data and runs a live prediction — proves the entry points work from a clean checkout, not just on my machine |
| **lint** | `ruff check` and `ruff format --check` — fails the build on unformatted code |
| **test** | `pytest` against the prediction path |
| **build-and-push** | Trains the model into the build context, builds the Docker image, and pushes to Docker Hub — gated on `needs: [test, lint]` |

The gating matters: **the image cannot publish unless the tests and the linter both pass.** A broken build stops at the boundary instead of shipping.

`models/` is git-ignored, so a fresh checkout has no trained artifact and the image's `COPY models/` had nothing to copy — the Docker build failed on every push for four months. The `build-and-push` job now trains the model into the build context first, so the image ships a model built from the committed data in the same run that tested it, rather than whatever binary happened to be committed.

The workflow is also **path-filtered** — it triggers only on changes to `src/`, `tests/`, `data/`, `requirements.txt`, or the workflow itself, so documentation edits don't burn CI minutes.

## Testing

[`tests/test_predict.py`](tests/test_predict.py) trains a fresh model in a module-scoped fixture and asserts behaviour through a parametrised sanity check, rather than loading a committed artifact — so the test covers the *training path* as well as prediction, and cannot pass against a stale model file.

```bash
pytest -v
```

## The Container

```bash
docker build -t sentiment-analysis-app .
docker run --rm sentiment-analysis-app
```

The [`Dockerfile`](Dockerfile) is ordered for **layer caching** — `requirements.txt` is copied and installed before the source is copied, so dependency layers are reused when only application code changes. Built on `python:3.11-slim` to keep the image small.

Dependencies are **fully pinned** in `requirements.txt` (`==`, not `>=`), so the image built by CI today is the image built six months from now.

## Running Locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train
python src/train.py --data data/sentiments.csv --out models/sentiment.joblib

# Predict
python src/predict.py --model models/sentiment.joblib "I loved this film" "That was bad"
```

Output is tab-separated `label<TAB>text`, with `1` for positive and `0` for negative.

## Structure

```
├── .github/workflows/ci.yml   # Four-job pipeline: predict, lint, test, build-and-push
├── Dockerfile                 # Cache-ordered, python:3.11-slim
├── pyproject.toml             # Ruff config: E, F, I, T20, UP
├── src/
│   ├── train.py               # Load → validate → stratified split → fit → save
│   └── predict.py             # Load model, score texts
├── tests/test_predict.py      # Trains a fresh model, asserts predictions
├── data/sentiments.csv        # 20 labelled sentences
└── models/sentiment.joblib    # Trained artifact
```

Ruff is configured (`pyproject.toml`) with `E`, `F`, `I`, `T20`, and `UP` — including `T20`, which bans stray `print` calls, so the deliberate user-facing ones carry explicit `# noqa: T201` markers rather than passing unnoticed.

Both scripts are `argparse` CLIs with typed signatures and docstrings. `train.py` validates that the input CSV carries `text` and `label` columns before doing any work, and **falls back to an unstratified split when stratification fails** — which it will on a dataset this small if a class ends up with too few members.

## What I'd Add Next

- **Model evaluation in CI** — the pipeline proves the code runs, but doesn't fail on a quality regression. A minimum-accuracy threshold would close that gap.
- **Image tagging by commit SHA** rather than only `latest`, so a deployment can be traced to a commit and rolled back.
- **A held-out test set that isn't the training data** — `train.py` splits internally, but there is no fixed evaluation set versioned alongside the model, so scores aren't comparable across runs.

## License

MIT
