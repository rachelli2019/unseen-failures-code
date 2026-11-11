# Unseen Failures

Research code, experiment configurations, and data for studying unseen failure modes and anytime-valid bounds.

## Paper version and updates

- **[Code accompanying arXiv v1](https://github.com/rachelli2019/unseen-failures-code/releases/tag/arxiv-v1)** is the initial paper snapshot.
- **[main](https://github.com/rachelli2019/unseen-failures-code)** contains subsequent fixes and improvements. Use the `arxiv-v1` tag to reproduce the initial release.

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

The initial release omits the two full processed datasets, `data/processed/swe_chat_sessions.parquet` and `data/processed/swe_chat_sessions_judge_ready_full.parquet`. Included taxonomy, population, labels, and split files support downstream experiments; workflows requiring the full session datasets need those files separately.

```bash
git clone https://github.com/rachelli2019/unseen-failures-code.git
cd unseen-failures-code
git checkout arxiv-v1
uv sync --frozen
```

Run commands from the repository root. To follow ongoing development afterward, switch back with `git checkout main`.

## Experiments

The [source guide](src/README.md) describes runners, method modules, and plotting tools. Configurations are in [`configs/`](configs/). For example:

```bash
uv run python src/run_primary_fixed.py --config configs/run_primary_oneshot_fixed.json
```

The supplied configuration runs 1,000 trials and may take substantial time. Generated intermediate results are written under `artifacts/` and are not included in this release.

Additional documentation:

- [Empirical Bernstein comparison](experiments/empirical_bernstein_headline/spec.md)
- [Imperfect-label comparison](experiments/imperfect_labels_reset/spec.md)
- [Data layout](data/README.md)
- [Figure and coverage catalog](data/plots/coverage_index.md)

Some retained experiment notes reference development-only files or historical local paths that are not included in this snapshot; use the included runners and configurations as the executable entry points.

## Tests

Build the native computation library first (requires a C++17 compiler):

```bash
mkdir -p artifacts/imperfect_labels_reset
c++ -O3 -std=c++17 -shared -fPIC src/weighted_restart_kernel.cpp -o artifacts/imperfect_labels_reset/weighted_restart.so
uv run python unit_tests/run_all.py
```

## Release maintenance

The `arxiv-v1` tag identifies the initial release and should not be moved. Publish cleanup and fixes as new commits on `main`; create new named releases for later paper versions.
