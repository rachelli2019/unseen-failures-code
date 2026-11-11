# Headline comparison with an empirical-Bernstein confidence sequence

**Experiment specification:** [spec.md](spec.md).
**Canonical figures:** [plot index](../../data/plots/full_population/primary_oneshot_fixed/empirical_bernstein/README.md).

This comparison supplements Figure 1a and Figure 3 of the current paper. It uses
the existing full-population primary-label trials: 4,961 sessions, 538 failure
modes, 1,000 trials, 3,000 draws, seed 888, and one-sided alpha = 0.05.
It does not use the superseded failure-only-population experiments.

## Baseline specification

Let Y_i be the exact-label novelty indicator, measured before updating history.
Its conditional mean is R_{i-1}. Howard et al. (2021), Theorem 4, gives an
empirical-Bernstein confidence sequence for the average conditional mean even
when those means vary. Monotonicity implies

    R_t <= (1/t) sum_{i=1}^t E[Y_i | F_{i-1}].

We use the one-sided lower-deviation martingale (or, equivalently, apply the
upper-deviation construction to 1-Y). The resulting upper bound is

    U_t = min(1, mean(Y_1,...,Y_t) + GE_alpha(V_t)/t),
    V_t = sum_{i=1}^t (Y_i - prediction_i)^2.

Predictions are the strictly past sample mean, with prediction_1 = 1/2.
GE is the gamma-exponential mixture boundary from Proposition 9, equation (65),
with c=1 and rho=100. The boundary is inverted numerically with an outward
rounding margin. rho=100 was fixed before computing these results; it is a
variance-time scale, not an empirically optimized choice. This comparison does
not establish robustness to all tuning choices or predictors.

The one-sided error budget is 0.05, not 0.025. The displayed two-sided Theorem 4
has error 2*alpha because it combines two tails; we only need one tail. Each
method independently receives the same nominal error budget. We do not select
the minimum bound across methods. No running intersection is applied to any
new baseline.

Source: https://arxiv.org/html/1810.08240v5#S4.SS1 and Proposition 9.

## Results

| Method | Mean UCB at t=3,000 | Empirical simultaneous coverage |
|---|---:|---:|
| Restart + K=100 lambda mixture (headline) | 0.102001 | 0.973 |
| Empirical-Bernstein CS, rho=100 | 0.108081 | 1.000 |
| No restart + same K=100 lambda mixture | 0.109773 | 0.971 |
| Simultaneous Good-Turing | 0.213314 | 1.000 |
| Pointwise Good-Turing (not licensed for optional stopping) | 0.150442 | 1.000 |

All coverage numbers are finite-simulation observations, not proofs of validity.
The methods have matched nominal guarantees, not matched empirical coverage.
In particular, EB is more conservative on these trajectories. The comparison
does not isolate historical-average lag from differences in coverage slack.

| Threshold | Mean stopping time, restart | Mean stopping time, EB | EB minus restart, paired 95% interval |
|---|---:|---:|---:|
| 0.12 | 2,152.209 | 2,276.925 | 124.716 [111.377, 138.055] |
| 0.13 | 1,743.857 | 1,811.339 | 67.482 [55.803, 79.161] |
| 0.14 | 1,370.116 | 1,438.206 | 68.090 [57.208, 78.972] |

All trials stop for both methods at these three thresholds. At epsilon=0.10,
41.2% of restart trials certify by 3,000 draws, versus 2.4% for EB and 0.9% for
the no-restart mixture. The respective means truncated at 3,000 are 2,895.127,
2,996.183, and 2,995.617; these are not uncensored mean stopping times.

The predetermined thresholds are 0.08, 0.09, ..., 0.14. We retain all trials and
report certification fraction and E[min(tau, 3000)]. Uncensored mean stopping
times are reported only when every trial stops. The CSV also reports premature
stopping probability over all trials. Paired intervals are normal-approximation
intervals across independent trials, not simultaneous bands across thresholds.

The headline method has a modest advantage over this EB specification on
certification cost, substantially smaller than the separation from Good-Turing.
The no-restart comparison is retained to distinguish the effect of restarting
within the same betting family.

## Figures and suggested paper integration

Outputs are in `data/plots/full_population/primary_oneshot_fixed/empirical_bernstein/`:

- `headline_bounds.pdf` / `.png`: replacement for the SWE-Chat bounds panel,
  with EB added to the existing baselines; no-restart is shown separately.
- `restart_ablation_bounds.pdf` / `.png`: restart versus no restart, with the oracle.
- `sequential_comparison.pdf` / `.png`: magnified bounds and certification
  fractions / restricted mean costs at the fixed thresholds.
- `coverage.pdf` / `.png`: main-comparison empirical simultaneous coverage.
- `restart_ablation_coverage.pdf` / `.png`: restart versus no-restart coverage.
- `summary.json`: configuration, input hashes, full source metadata, and results.
- `stopping.csv`: per-method results across thresholds, including censored trials.

Bound lines show means, with empirical 2.5th–97.5th percentile bands across
trials. These bands show trajectory variation, not confidence intervals for the
mean. The magnified comparison panel clips early bounds above 0.25 for visibility;
the full headline panel shows the entire [0,1] range. Stopping curves show all
trials with censoring handled as described above. Synthetic coverage in the
paper's Figure 1b is unchanged; this run only adds the SWE-Chat comparison.

Suggested replacement caption for the SWE-Chat panel:

> Upper bounds on unseen-failure probability on SWE-Chat, comparing the restart
> mixture with an empirical-Bernstein confidence sequence and two Good-Turing bounds. Curves show means and
> shading shows empirical 2.5th–97.5th percentiles across 1,000 paired trials.
> All methods use alpha=0.05; the dashed pointwise Good-Turing bound does not
> provide an anytime-valid guarantee. At 3,000 draws, mean upper bounds are
> 0.1020 for the restart mixture and 0.1081 for empirical-Bernstein. Their
> empirical simultaneous coverages are 0.973 and 1.000, respectively.

The supplied paper PDF is not edited; these are replacement figures and text
for integration into its authoring source.

## Reproduction and validation

Run from the repository root with the existing NumPy/SciPy/Matplotlib environment:

```sh
.venv/bin/python -m unittest unit_tests.test_empirical_bernstein
.venv/bin/python src/run_empirical_bernstein_headline.py
```

Configuration: `configs/run_empirical_bernstein_headline.json`.
Numerics: `src/empirical_bernstein.py`.
The runner checks confidence levels, seeds, trial counts, horizon, oracle
trajectories, and (where available) exact novelty trajectories before pairing.
The GT artifact lacks novelty indicators, so its pairing is checked against
metadata and the full oracle matrix. Output trajectories are saved under
`artifacts/empirical_bernstein/primary_fixed_fullpop/trajectories.npz`.

Tests cover independent numerical quadrature of the mixture formula for four
rho values, boundary inversion, predictable variance computation, prefix
invariance, exact finite-species coverage enumeration, input validation, and
strict stopping/censoring semantics.
