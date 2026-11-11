# Unseen Failures

Code and experiment configurations for **Controlling the Probability of Unknown Unknowns in Pre-Deployment Testing of AI Systems**.

## Getting started

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --frozen
uv run python src/run_primary_fixed.py --config configs/run_primary_oneshot_fixed.json
```

See [`configs/`](configs/) for experiment settings and [`src/README.md`](src/README.md) for the runners and plotting tools.
