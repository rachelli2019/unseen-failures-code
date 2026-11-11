# src/

Two families of run script live here. They differ in what they produce, not in what they are
allowed to run on.

**Headline runners** — one per setting, producing that setting's bound *and its baselines*. The
baselines are not interchangeable across settings, which is why these are separate scripts rather
than one parameterised one: Good-Turing needs a species/singleton structure, so set-valued has no
GT arrays at all, and the online rule's GT is a heuristic over a greedy-cover taxonomy whose arrays
mean something different from primary's Bonferroni pair.

| Script | Setting | Baseline arrays |
|---|---|---|
| `run_primary_fixed.py` | fixed taxonomy, one mode per session | `gt_hat`, `gt_upper_bound`, `gt_bonferroni_bound` |
| `run_primary_online.py` | online pairwise rule | `gt_hat`, `gt_bound`, `gt_bound_uncorrected` |
| `run_set_valued_fixed.py` | fixed taxonomy, mode set per session | none |
| `run_bernoulli_simulation.py` | i.i.d. Bernoulli(p) stress test | none; different npz shape |

**λ-mixture runners** — the bound only, on *any* setting. Here discovery genuinely is the only
difference, so it is abstracted into `lambda_grid_discovery_backends.py` and selected by a
`discovery` block in the config.

| Script | Produces |
|---|---|
| `run_lambda_grid_mixture.py` | one bound per configured arm; an arm is a set of λ with weights, so fixed λ, a K-point grid and anything between are all expressible |

`run_lambda_grid_mixture.py`'s `single_lambda_0p25` arm is the same quantity as the headline
runners' `restart_mixture_everytime_heavytail_matrix` — verified to 9.5e-07, which is the 20-vs-50
bisection-iteration difference under outward rounding. The two families overlap on the bound and
differ on the baselines.

**Which to use.** Adding a λ treatment or a setting → the λ-mixture runners. Changing a baseline, or
regenerating a published headline figure → the headline runner for that setting.

## Method modules

| | |
|---|---|
| `av_core.py` | e-process, restart schedules and weights, restart-mixture UCB, classical Good-Turing bound |
| `av_lambda_restart_mixture.py` | mixture over a λ grid; `everytime_*` is the fast path used in production, the general `lambda_restart_mixture_*` pair is the reference implementation it is tested against |
| `av_closed_form_lambda_mixture.py` | hindsight-optimal λ in closed form (KL form); not anytime-valid, see its docstring |
| `av_discovery_fixed_primary.py`, `av_discovery_fixed_set_valued.py`, `av_discovery_online_primary.py` | z/q bookkeeping per representation; the λ backends delegate here rather than reimplementing |
| `av_evaluation_metrics.py` | coverage, gap-to-truth, stopping times |
| `lambda_grid_discovery_backends.py` | `discovery` block → `{prepare, draw, describe}` for the four settings |
| `run_config.py` | JSON config loading; requires exactly the declared keys, so a typo raises |

## Sharding

`run_lambda_grid_mixture.py` takes `--task-index` / `--n-tasks` and
writes one shard per task; `combine_lambda_grid_shards.py` merges them, checking every trial index
appears exactly once. `combine_bernoulli_shards.py` does the same for `run_bernoulli_simulation.py`.

Trial *i* always uses `SeedSequence(seed).spawn(n_trials)[i]`. Children depend only on index, so
shards concatenate to exactly one unsharded run, a run can be extended later without recomputing,
and every runner at the same seed sees the same trials.

## Diagnostics

`run_lambda_restart_ablation.py` (λ and restart-schedule ablation),
`oracle_best_restart_diagnostic.py` (how much the mixture loses to a hindsight restart time).
Separate from the headline path; not inputs to any published figure.

`run_pac_labeling.py` / `pac_labeling_core.py` are unrelated to the bound — PAC safety labelling.

## Plotting

`plotting/` reads the npz files the runners write and does no recomputation.
`plot_av_pipeline.py` renders the headline figure and takes `--ours-npz` to swap in a bound from a
different run. `plot_lambda_comparison.py` renders the four-arm λ comparison. Scripts that combine
runs check the oracle trajectories agree and refuse to plot otherwise, so methods scored on
different draws cannot be silently compared.

## Documentation

### Coverage catalog

Run `python src/plotting/build_coverage_index.py` after generating plots to rebuild
`data/plots/coverage_index.csv` and `data/plots/coverage_index.md`. The Markdown report links
each figure and its supporting report. The CSV has one row per figure/method, with source
run, trajectory file, array key, trial count, evaluated horizon, violation count, and final
empirical simultaneous coverage. PNG/PDF pairs count as one figure. Different figures can
reuse the same run, so rows must not be pooled as independent trials.

Coverage means no violation at **any** timestep through the recorded horizon. The pipeline
plotter evaluates its requested prefix, which can be shorter than the saved run. Original
violation counts take precedence over rounded Markdown percentages. Missing historical
source/target fields stay blank; figures without reports are listed explicitly.

Pipeline, Bernoulli, ablation and separate-panel plotters now write `*_coverage_record.json`
beside each plot. Comparison plotters retain their summaries and record per-arm source keys.
The catalog prefers explicit records over old Markdown reports. Preserve records alongside
their figures, and use a new output prefix/directory to retain multiple run versions.
The catalog describes files currently on disk; overwritten historical figures cannot be recovered.

Method and results: `experimental_writeup/statistical_evaluation.md` (headline),
`experimental_writeup/lambda_mixture.md` (λ treatments). Per-run commands and status:
`experiments/<slug>/log.md`.

### Final imperfect-label comparison

Use `run_imperfect_labels_reset.py` for the final perfect-label, IPW B-PAC,
and prediction-powered (r=0) experiment. It is distinct from the older
`run_pac_labeling.py` implementation. See
[the specification](../experiments/imperfect_labels_reset/spec.md) for formulas,
input provenance and reproduction commands, and
[the results](../experiments/imperfect_labels_reset/reports_1000/results.md)
for the 1,000-trial comparison. `plotting/plot_imperfect_labels_paper.py` renders
bounds, coverage, and total ground-truth labels through stopping.
