# Final empirical simultaneous coverage

87 method entries across 30 figures. PNG/PDF pairs count as one figure.

Coverage is the fraction of trials with no violation at any timestep through the evaluated horizon. Rows from different figures may reuse the same trials; do not pool them. Blank CSV fields and ‘unknown’ below mean the historical report did not record that detail. Target coverage is not inferred from a plot's confidence-band level or assumed to be 95%.

Refresh from the repository root: `python src/plotting/build_coverage_index.py`.

[Download the complete CSV](coverage_index.csv). The CSV includes source paths, array keys and provenance.

## Full population


### primary_llm_merge_fixed_taxonomy_fullpop_bounds

[Plot](full_population/primary_llm_merge_fixed/primary_llm_merge_fixed_taxonomy_fullpop_bounds.png) · [Source report](full_population/primary_llm_merge_fixed/primary_llm_merge_fixed_taxonomy_fullpop_coverage.md)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| Anytime-Valid UCB (ours) | 99.70% | 3 | unknown |
| GT UCB (pointwise) | 100.00% | 0 | unknown |
| GT UCB (simultaneous) | 100.00% | 0 | unknown |

### primary_llm_merge_fixed_taxonomy_fullpop_lambda_grid_bounds

[Plot](full_population/primary_llm_merge_fixed/primary_llm_merge_fixed_taxonomy_fullpop_lambda_grid_bounds.png) · [Source report](full_population/primary_llm_merge_fixed/primary_llm_merge_fixed_taxonomy_fullpop_lambda_grid_coverage.md)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| Anytime-Valid UCB (ours) | 97.90% | 21 | unknown |
| GT UCB (pointwise) | 100.00% | 0 | unknown |
| GT UCB (simultaneous) | 100.00% | 0 | unknown |

### primary_fixed_fullpop_grid_vs_fixed_lambda

[Plot](full_population/primary_oneshot_fixed/primary_fixed_fullpop_grid_vs_fixed_lambda.png) · [Source report](full_population/primary_oneshot_fixed/primary_fixed_fullpop_grid_vs_fixed_lambda_summary.json)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| $\lambda$ grid, $K=100$ | 97.30% | 27 | primary_fixed_fullpop_s888_K100 |
| Fixed $\lambda = 0.1$ | 100.00% | 0 | primary_fixed_fullpop_s888_extra_arms |
| Fixed $\lambda = 0.25$ | 99.20% | 8 | primary_fixed_fullpop_s888_K100 |
| Fixed $\lambda = 0.5$ | 96.60% | 34 | primary_fixed_fullpop_s888_extra_arms |

### primary_fixed_fullpop_lambda_comparison

[Plot](full_population/primary_oneshot_fixed/primary_fixed_fullpop_lambda_comparison.png) · [Source report](full_population/primary_oneshot_fixed/primary_fixed_fullpop_lambda_comparison_summary.json)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| $\lambda$ grid, $K=100$ | 97.30% | 27 | primary_fixed_fullpop_s888_K100 |
| Best in hindsight (cheating) | 89.70% | 103 | primary_fixed_fullpop_s888_cf |
| Closed-form $\lambda$ | 99.80% | 2 | primary_fixed_fullpop_s888_cf |
| Fixed $\lambda = 0.25$ | 99.20% | 8 | primary_fixed_fullpop_s888_K100 |

### primary_fixed_fullpop_restart_vs_norestart

[Plot](full_population/primary_oneshot_fixed/primary_fixed_fullpop_restart_vs_norestart.png) · [Source report](full_population/primary_oneshot_fixed/primary_fixed_fullpop_restart_vs_norestart_summary.json)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| No restart (single process) | 97.10% | 29 | primary_fixed_fullpop_s888_extra_arms |
| Restart mixture (every-time) | 97.30% | 27 | primary_fixed_fullpop_s888_K100 |

### primary_oneshot_fixed_taxonomy_fullpop_bounds

[Plot](full_population/primary_oneshot_fixed/primary_oneshot_fixed_taxonomy_fullpop_bounds.png) · [Source report](full_population/primary_oneshot_fixed/primary_oneshot_fixed_taxonomy_fullpop_coverage.md)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| Anytime-Valid UCB (ours) | 99.20% | 8 | unknown |
| GT UCB (pointwise) | 100.00% | 0 | unknown |
| GT UCB (simultaneous) | 100.00% | 0 | unknown |

### primary_oneshot_fixed_taxonomy_fullpop_lambda_grid_bounds

[Plot](full_population/primary_oneshot_fixed/primary_oneshot_fixed_taxonomy_fullpop_lambda_grid_bounds.png) · [Source report](full_population/primary_oneshot_fixed/primary_oneshot_fixed_taxonomy_fullpop_lambda_grid_coverage.md)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| Anytime-Valid UCB (ours) | 97.30% | 27 | unknown |
| GT UCB (pointwise) | 100.00% | 0 | unknown |
| GT UCB (simultaneous) | 100.00% | 0 | unknown |

### primary_oneshot_online_taxonomy_fullpop_bounds

[Plot](full_population/primary_oneshot_online/primary_oneshot_online_taxonomy_fullpop_bounds.png) · [Source report](full_population/primary_oneshot_online/primary_oneshot_online_taxonomy_fullpop_coverage.md)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| Anytime-Valid UCB (ours) | 99.60% | 4 | unknown |
| GT UCB (pointwise) | 100.00% | 0 | unknown |
| GT UCB (simultaneous) | 100.00% | 0 | unknown |

### primary_oneshot_online_taxonomy_fullpop_lambda_grid_bounds

[Plot](full_population/primary_oneshot_online/primary_oneshot_online_taxonomy_fullpop_lambda_grid_bounds.png) · [Source report](full_population/primary_oneshot_online/primary_oneshot_online_taxonomy_fullpop_lambda_grid_coverage.md)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| Anytime-Valid UCB (ours) | 98.10% | 19 | unknown |
| GT UCB (pointwise) | 100.00% | 0 | unknown |
| GT UCB (simultaneous) | 100.00% | 0 | unknown |

### set_valued_fullpop_grid_vs_fixed_lambda

[Plot](full_population/set_valued_oneshot_fixed/set_valued_fullpop_grid_vs_fixed_lambda.png) · [Source report](full_population/set_valued_oneshot_fixed/set_valued_fullpop_grid_vs_fixed_lambda_summary.json)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| $\lambda$ grid, $K=100$ | 97.50% | 25 | set_valued_fullpop_s888_K100 |
| Fixed $\lambda = 0.1$ | 100.00% | 0 | set_valued_fullpop_s888_extra_arms |
| Fixed $\lambda = 0.25$ | 99.00% | 10 | set_valued_fullpop_s888_K100 |
| Fixed $\lambda = 0.5$ | 96.40% | 36 | set_valued_fullpop_s888_extra_arms |

### set_valued_fullpop_lambda_comparison

[Plot](full_population/set_valued_oneshot_fixed/set_valued_fullpop_lambda_comparison.png) · [Source report](full_population/set_valued_oneshot_fixed/set_valued_fullpop_lambda_comparison_summary.json)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| $\lambda$ grid, $K=100$ | 97.50% | 25 | set_valued_fullpop_s888_K100 |
| Best in hindsight (cheating) | 89.70% | 103 | set_valued_fullpop_s888_cf |
| Closed-form $\lambda$ | 99.90% | 1 | set_valued_fullpop_s888_cf |
| Fixed $\lambda = 0.25$ | 99.00% | 10 | set_valued_fullpop_s888_K100 |

### set_valued_fullpop_restart_vs_norestart

[Plot](full_population/set_valued_oneshot_fixed/set_valued_fullpop_restart_vs_norestart.png) · [Source report](full_population/set_valued_oneshot_fixed/set_valued_fullpop_restart_vs_norestart_summary.json)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| No restart (single process) | 96.90% | 31 | set_valued_fullpop_s888_extra_arms |
| Restart mixture (every-time) | 97.50% | 25 | set_valued_fullpop_s888_K100 |

### set_valued_oneshot_fixed_taxonomy_fullpop_bounds

[Plot](full_population/set_valued_oneshot_fixed/set_valued_oneshot_fixed_taxonomy_fullpop_bounds.png) · [Source report](full_population/set_valued_oneshot_fixed/set_valued_oneshot_fixed_taxonomy_fullpop_coverage.md)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| Anytime-Valid UCB (ours) | 99.00% | 10 | unknown |

### set_valued_oneshot_fixed_taxonomy_fullpop_lambda_grid_bounds

[Plot](full_population/set_valued_oneshot_fixed/set_valued_oneshot_fixed_taxonomy_fullpop_lambda_grid_bounds.png) · [Source report](full_population/set_valued_oneshot_fixed/set_valued_oneshot_fixed_taxonomy_fullpop_lambda_grid_coverage.md)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| Anytime-Valid UCB (ours) | 97.50% | 25 | unknown |

## Other and legacy plots


### bernoulli_coverage

[Plot](bernoulli/bernoulli_coverage.png) · [Source report](bernoulli/bernoulli_coverage_coverage_record.json)

Trials: 10,000. Evaluated horizon: 10,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| Anytime-Valid UCB (ours) | 95.27% | 473 | h10000_t10000 |

### bernoulli_lambda_comparison

[Plot](lambda_comparison/bernoulli_lambda_comparison.png) · [Source report](lambda_comparison/bernoulli_lambda_comparison_summary.json)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| $\lambda$ grid, $K=100$ | 96.70% | 33 | bernoulli_s888_K100_v2 |
| Best in hindsight (cheating) | 89.20% | 108 | bernoulli_s888_cf2 |
| Closed-form $\lambda$ | 100.00% | 0 | bernoulli_s888_cf2 |
| Fixed $\lambda = 0.25$ | 96.30% | 37 | bernoulli_s888_K100_v2 |

### online_lambda_comparison

[Plot](lambda_comparison/online_lambda_comparison.png) · [Source report](lambda_comparison/online_lambda_comparison_summary.json)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| $\lambda$ grid, $K=100$ | 100.00% | 0 | online_s888_K100_v2 |
| Best in hindsight (cheating) | 98.30% | 17 | online_s888_cf2 |
| Closed-form $\lambda$ | 99.80% | 2 | online_s888_cf2 |
| Fixed $\lambda = 0.25$ | 100.00% | 0 | online_s888_K100_v2 |

### primary_fixed_lambda_comparison

[Plot](lambda_comparison/primary_fixed_lambda_comparison.png) · [Source report](lambda_comparison/primary_fixed_lambda_comparison_summary.json)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| $\lambda$ grid, $K=100$ | 100.00% | 0 | primary_fixed_s888_K100 |
| Best in hindsight (cheating) | 98.20% | 18 | primary_fixed_s888_cf2 |
| Closed-form $\lambda$ | 99.90% | 1 | primary_fixed_s888_cf2 |
| Fixed $\lambda = 0.25$ | 100.00% | 0 | lambda_restart_ablation |

### set_valued_lambda_comparison

[Plot](lambda_comparison/set_valued_lambda_comparison.png) · [Source report](lambda_comparison/set_valued_lambda_comparison_summary.json)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| $\lambda$ grid, $K=100$ | 100.00% | 0 | set_valued_s888_K100_v2 |
| Best in hindsight (cheating) | 98.50% | 15 | set_valued_s888_cf2 |
| Closed-form $\lambda$ | 99.90% | 1 | set_valued_s888_cf2 |
| Fixed $\lambda = 0.25$ | 100.00% | 0 | set_valued_s888_K100_v2 |

### primary_fixed_K100_bounds

[Plot](lambda_grid_mixture/primary_fixed_K100_bounds.png) · [Source report](lambda_grid_mixture/primary_fixed_K100_summary.json)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| Single λ = 0.1 | 100.00% | 0 | lambda_restart_ablation |
| Single λ = 0.25 | 100.00% | 0 | lambda_restart_ablation |
| Single λ = 0.5 | 100.00% | 0 | lambda_restart_ablation |
| Uniform λ grid, K = 100 | 100.00% | 0 | primary_fixed_s888_K100 |

### primary_fixed_K100_ratio

[Plot](lambda_grid_mixture/primary_fixed_K100_ratio.png) · [Source report](lambda_grid_mixture/primary_fixed_K100_summary.json)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| Single λ = 0.1 | 100.00% | 0 | lambda_restart_ablation |
| Single λ = 0.25 | 100.00% | 0 | lambda_restart_ablation |
| Single λ = 0.5 | 100.00% | 0 | lambda_restart_ablation |
| Uniform λ grid, K = 100 | 100.00% | 0 | primary_fixed_s888_K100 |

### primary_oneshot_lambda_restart_ablation_lambda_bounds

[Plot](lambda_restart_ablation/primary_oneshot_lambda_restart_ablation_lambda_bounds.png) · [Source report](lambda_restart_ablation/primary_oneshot_lambda_restart_ablation_lambda_bounds_coverage_record.json)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| Restart mixture, $\lambda$ = 0.1 | 100.00% | 0 | lambda_restart_ablation |
| Restart mixture, $\lambda$ = 0.25 | 100.00% | 0 | lambda_restart_ablation |
| Restart mixture, $\lambda$ = 0.5 | 100.00% | 0 | lambda_restart_ablation |

### primary_oneshot_lambda_restart_ablation_restart_bounds

[Plot](lambda_restart_ablation/primary_oneshot_lambda_restart_ablation_restart_bounds.png) · [Source report](lambda_restart_ablation/primary_oneshot_lambda_restart_ablation_restart_bounds_coverage_record.json)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| No restart | 100.00% | 0 | lambda_restart_ablation |
| Restart mixture | 100.00% | 0 | lambda_restart_ablation |

### primary_llm_merge_fixed_taxonomy_bounds

[Plot](primary_llm_merge_fixed/primary_llm_merge_fixed_taxonomy_bounds.png) · [Source report](primary_llm_merge_fixed/primary_llm_merge_fixed_taxonomy_coverage.md)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| Anytime-Valid UCB (ours) | 100.00% | 0 | unknown |
| GT UCB (pointwise) | 100.00% | 0 | unknown |
| GT UCB (simultaneous) | 100.00% | 0 | unknown |

### primary_oneshot_fixed_taxonomy_bounds

[Plot](primary_oneshot_fixed/primary_oneshot_fixed_taxonomy_bounds.png) · [Source report](primary_oneshot_fixed/primary_oneshot_fixed_taxonomy_coverage.md)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| Anytime-Valid UCB (ours) | 100.00% | 0 | unknown |
| GT UCB (pointwise) | 100.00% | 0 | unknown |
| GT UCB (simultaneous) | 100.00% | 0 | unknown |

### primary_oneshot_fixed_taxonomy_lambda_grid_bounds

[Plot](primary_oneshot_fixed/primary_oneshot_fixed_taxonomy_lambda_grid_bounds.png) · [Source report](primary_oneshot_fixed/primary_oneshot_fixed_taxonomy_lambda_grid_coverage.md)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| Anytime-Valid UCB (ours) | 100.00% | 0 | unknown |
| GT UCB (pointwise) | 100.00% | 0 | unknown |
| GT UCB (simultaneous) | 100.00% | 0 | unknown |

### primary_oneshot_online_taxonomy_bounds

[Plot](primary_oneshot_online/primary_oneshot_online_taxonomy_bounds.png) · [Source report](primary_oneshot_online/primary_oneshot_online_taxonomy_coverage.md)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| Anytime-Valid UCB (ours) | 100.00% | 0 | unknown |
| GT UCB (pointwise) | 100.00% | 0 | unknown |
| GT UCB (simultaneous) | 100.00% | 0 | unknown |

### primary_oneshot_online_taxonomy_lambda_grid_bounds

[Plot](primary_oneshot_online/primary_oneshot_online_taxonomy_lambda_grid_bounds.png) · [Source report](primary_oneshot_online/primary_oneshot_online_taxonomy_lambda_grid_coverage.md)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| Anytime-Valid UCB (ours) | 100.00% | 0 | unknown |
| GT UCB (pointwise) | 100.00% | 0 | unknown |
| GT UCB (simultaneous) | 100.00% | 0 | unknown |

### set_valued_oneshot_fixed_taxonomy_bounds

[Plot](set_valued_oneshot_fixed/set_valued_oneshot_fixed_taxonomy_bounds.png) · [Source report](set_valued_oneshot_fixed/set_valued_oneshot_fixed_taxonomy_coverage.md)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| Anytime-Valid UCB (ours) | 100.00% | 0 | unknown |

### set_valued_oneshot_fixed_taxonomy_lambda_grid_bounds

[Plot](set_valued_oneshot_fixed/set_valued_oneshot_fixed_taxonomy_lambda_grid_bounds.png) · [Source report](set_valued_oneshot_fixed/set_valued_oneshot_fixed_taxonomy_lambda_grid_coverage.md)

Trials: 1,000. Evaluated horizon: 3,000.

| Method | Coverage | Violating trials | Source run |
|---|---:|---:|---|
| Anytime-Valid UCB (ours) | 100.00% | 0 | unknown |

## Figures without coverage records

- [primary_fixed_fullpop_grid_vs_fixed_bounds](full_population/primary_oneshot_fixed/primary_fixed_fullpop_grid_vs_fixed_bounds.png)
- [primary_fixed_fullpop_grid_vs_fixed_coverage](full_population/primary_oneshot_fixed/primary_fixed_fullpop_grid_vs_fixed_coverage.png)
- [primary_fixed_fullpop_restart_vs_norestart_panel_bounds](full_population/primary_oneshot_fixed/primary_fixed_fullpop_restart_vs_norestart_panel_bounds.png)
- [primary_fixed_fullpop_restart_vs_norestart_panel_coverage](full_population/primary_oneshot_fixed/primary_fixed_fullpop_restart_vs_norestart_panel_coverage.png)
