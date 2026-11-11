# Reproducing the experiments

Run these commands from the repository root after `uv sync --frozen`. Configurations specify the seeds, trial counts, and output paths. Full runs may take substantial time.

## Primary comparison

Generate the proposed bound, no-restart ablation, and Good–Turing baselines, then render the comparison with empirical Bernstein bounds:

```bash
uv run python src/run_primary_fixed.py --config configs/run_primary_oneshot_fixed.json
uv run python src/run_lambda_grid_mixture.py --config configs/run_lambda_grid_mixture_primary_fixed.json --task-index 0 --n-tasks 1 --output-dir artifacts/lambda_grid_mixture/primary_fixed_fullpop_s888_K100
uv run python src/combine_lambda_grid_shards.py --shard-dir artifacts/lambda_grid_mixture/primary_fixed_fullpop_s888_K100 --output-npz artifacts/lambda_grid_mixture/primary_fixed_fullpop_s888_K100/combined/trajectories.npz
uv run python src/run_lambda_grid_mixture.py --config configs/run_lambda_grid_extra_arms_primary_fixed.json --task-index 0 --n-tasks 1 --output-dir artifacts/lambda_grid_mixture/primary_fixed_fullpop_s888_extra_arms
uv run python src/combine_lambda_grid_shards.py --shard-dir artifacts/lambda_grid_mixture/primary_fixed_fullpop_s888_extra_arms --output-npz artifacts/lambda_grid_mixture/primary_fixed_fullpop_s888_extra_arms/combined/trajectories.npz
uv run python src/run_empirical_bernstein_headline.py
```

Figures and summary statistics are written to `data/plots/full_population/primary_oneshot_fixed/empirical_bernstein/`.

## Other settings

| Setting | Runner | Configuration |
|---|---|---|
| Online categorization | `run_primary_online.py` | `run_primary_online_oneshot.json` |
| Set-valued categorization | `run_set_valued_fixed.py` | `run_set_valued_oneshot_fixed.json` |
| LLM-merged taxonomy | `run_primary_fixed.py` | `run_primary_llm_merge_fixed.json` |
| Nonuniform sampling | `run_fixed_q_primary.py` | `run_fixed_q_primary.json` |
| Restart ablation | `run_lambda_restart_ablation.py` | `run_lambda_restart_ablation.json` |

Use `uv run python src/<runner> --config configs/<configuration>`. The nonuniform-sampling runner requires a C++17 compiler and the processed session dataset specified in its configuration.

The `run_lambda_grid_mixture_*.json` and `run_lambda_grid_extra_arms_*.json` configurations use the same mixture runner and shard-combining commands as above, with a separate output directory per run. The Bernoulli configurations generate synthetic observations without session data. Plotting scripts in `src/plotting/` accept `--help` for their input and output arguments, except the fixed imperfect-label plotting command below.

## Imperfect labels

Build the native library with a C++17 compiler, then run the comparison and complete the stopping-time calculation:

```bash
mkdir -p artifacts/imperfect_labels_reset
c++ -O3 -std=c++17 -shared -fPIC src/weighted_restart_kernel.cpp -o artifacts/imperfect_labels_reset/weighted_restart.so
uv run python src/run_imperfect_labels_reset.py --config configs/run_imperfect_labels_final.json
uv run python src/run_imperfect_labels_reset.py --config configs/run_imperfect_labels_ppi_continuation.json
uv run python src/complete_ppi_stopping.py
uv run python src/plotting/plot_imperfect_labels_paper.py
```

Figures are written to `data/plots/imperfect_labels/`. The saved inputs are described in [`data/imperfect_labels/README.md`](../data/imperfect_labels/README.md).

## Implementation

- `av_core.py` and `av_lambda_restart_mixture.py`: e-processes, restart mixtures, and upper bounds.
- `av_discovery_*.py`: discovery rules for each categorization setting.
- `empirical_bernstein.py`: variance-adaptive comparison bound.
- `imperfect_labels_reset_core.py`, `av_weighted_restart.py`, and `weighted_restart_kernel.cpp`: imperfect-label methods and native computation.
