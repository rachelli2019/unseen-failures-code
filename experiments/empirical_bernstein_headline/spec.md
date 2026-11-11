# Fixed-taxonomy headline comparison: restarts and empirical-Bernstein

## Purpose

Measure (1) the contribution of restarting within the existing betting method
and (2) performance against an established variance-adaptive confidence sequence.
This extends the SWE-Chat comparison in Figures 1a and 3 of the current paper.

## Shared experimental setup

| Setting | Specification |
|---|---|
| Population | The original full test population of 4,961 SWE-Chat sessions |
| Novelty rule | Fixed primary-label taxonomy with 538 failure modes |
| Labels | Treated as exact for this experiment |
| Sampling | Independent uniform draws with replacement |
| Replications | 1,000 paired trials; seed 888 |
| Horizon | 3,000 draws per trial |
| Error level | One-sided alpha = 0.05 for each method |
| Input data | Reuse saved trajectories; no new sampling or labeling |

At draw i, before updating the observed history, record

\[
Y_i=\mathrm{fail}(A_i)\,\mathrm{unseen}(A_i,H_{i-1}).
\]

Historical indicators are never relabeled after later discoveries. The oracle
\(R_t\) is the probability, under the empirical population, that the next draw
contains an unseen failure after updating history through draw t. Thus
\(\mathbb E[Y_i\mid\mathcal F_{i-1}]=R_{i-1}\) and \(R_t\) is nonincreasing.
Oracle values are used only for evaluation, not for constructing the bounds.

## Methods

### 1. Original restart mixture

Retain the saved headline bound without changing its implementation. It averages
betting processes over every restart time and a fixed grid
\(\lambda_k=(k-\tfrac12)/100\), k = 1,...,100, with equal lambda weights.
Each component is

\[
E_{s:t}^{\lambda}(u)=\prod_{i=s}^{t}[1+\lambda(u-Y_i)].
\]

The implementation uses log-squared restart weights
\(w_s=[2.11(s+1)\log^2(s+1)]^{-1}\), with unstarted reserve
\(0.999-\sum_{s\le t}w_s\). Average component wealth before inversion at
\(1/\alpha\). Reuse `bound__uniform_K100` from the headline artifact.

### 2. No-restart ablation

Keep the same lambda grid, equal lambda weights, observations, alpha, and
inversion convention. Remove only the mixture over restart times:

\[
M_t^{\mathrm{NR}}(u)=\frac1{100}\sum_{k=1}^{100}E_{1:t}^{\lambda_k}(u),
\qquad
U_t^{\mathrm{NR}}=\sup\{u\in(0,1):M_t^{\mathrm{NR}}(u)<1/\alpha\}.
\]

Every component starts at the first observation; there is no restart reserve.
Reuse `bound__uniform_K100_norestart` from the saved ablation artifact.

### 3. Empirical-Bernstein confidence sequence

Use the one-sided construction underlying Howard et al. (2021), Theorem 4,
with observations in [0,1]. Set

\[
\widehat Y_1=\tfrac12,\qquad
\widehat Y_i=\frac1{i-1}\sum_{j<i}Y_j,\qquad
V_t=\sum_{i=1}^{t}(Y_i-\widehat Y_i)^2.
\]

Construct

\[
U_t^{\mathrm{EB}}=\min\{1,\bar Y_t+u_\alpha(V_t)/t\}.
\]

Here \(u_\alpha\) is the gamma-exponential mixture boundary in Proposition 9,
equation (65), with scale c=1 and rho=100. Evaluate its log-mixture using
log-gamma and the regularized lower incomplete gamma function; bracket and solve
\(\log m(s,v)=\log(1/\alpha)\) by 40 bisection steps. Return the upper bracket
plus 1e-10 as an outward numerical margin. rho=100 was fixed before examining
these comparison results; it is not fitted to the trajectories.

The relevant tail bounds the conditional-mean sum minus the observed sum.
Equivalently, apply the upper-deviation construction to 1-Y, predicting
1-minus-the-predictor above; the squared prediction errors are unchanged.
The screenshot's two-sided Theorem 4 has error 2*alpha because it combines
two tails. This implementation uses only the required tail at alpha=0.05.

The confidence sequence covers the average conditional mean, which in turn
bounds current unseen-failure probability:

\[
R_t\le\frac1t\sum_{i=1}^{t}\mathbb E[Y_i\mid\mathcal F_{i-1}]
\le U_t^{\mathrm{EB}}
\quad\text{simultaneously with probability at least }0.95.
\]

Reference: [Howard et al., Theorem 4 and Proposition 9](https://arxiv.org/html/1810.08240v5).

### Existing context baselines

Retain both saved Good-Turing curves. The simultaneous version uses the existing
Bonferroni correction over 3,000 draws. The pointwise version is displayed as
context only and excluded from stopping comparisons.

Do not combine methods by selecting their smallest bound. Do not choose lambda,
rho, or predictors based on test results. The new EB baseline uses no restarts
or running intersection. Nominal error levels are matched; observed coverages
need not match.

## Evaluation

1. **Bound tightness:** mean bound through time, empirical 2.5th–97.5th percentile
   bands across trials, and mean at draw 3,000. These bands describe trajectory
   variation, not uncertainty in the estimated mean.
2. **Simultaneous coverage:** fraction of trials satisfying
   \(R_s\le U_s\) for every s through each plotted time t.
3. **Certification:** use strict stopping
   \(\tau(\varepsilon)=\inf\{t:U_t<\varepsilon\}\), at the fixed thresholds
   0.08, 0.09, 0.10, 0.11, 0.12, 0.13, and 0.14.
4. **Censoring:** retain every trial. Report the fraction certified by 3,000 and
   the restricted mean \(\operatorname{mean}(\min\{\tau,3000\})\). Report an
   uncensored sample mean only when every trial stops within the horizon.
5. **Paired differences:** comparator minus restart restricted cost, with
   descriptive 95% intervals given by paired mean +/- 1.96 paired standard errors.
   These intervals are not simultaneous over thresholds.
6. **Premature stopping:** fraction of all trials that stop within the horizon
   with \(R_\tau\ge\varepsilon\).

The practical stopping threshold is epsilon. This experiment does not directly
test the theorem's delay after crossing a lower gamma < epsilon, nor isolate
historical-average lag from differences in statistical conservatism.

## Reproduction and checks

Configuration: [run_empirical_bernstein_headline.json](../../configs/run_empirical_bernstein_headline.json).
Input artifact paths and all settings are specified there. From the project root:

```sh
.venv/bin/python -m unittest unit_tests.test_empirical_bernstein
.venv/bin/python src/run_empirical_bernstein_headline.py
```

The runner verifies alpha, seed, trial count, horizon, and full oracle matrices;
it additionally verifies exact novelty matrices for the restart/no-restart
pair. Good-Turing's saved artifact has no novelty matrix. The output records
source hashes and metadata for provenance.

Validation covers the mixture against independent quadrature, boundary
inversion, strictly past prediction and prefix invariance, finite-species
coverage enumeration, input validation, and strict stopping/censoring semantics.

## Canonical outputs

Keep publication figures together in
[`data/plots/full_population/primary_oneshot_fixed/empirical_bernstein/`](../../data/plots/full_population/primary_oneshot_fixed/empirical_bernstein/README.md):

- **`headline_bounds.png` / `.pdf`:** main comparison of ours against EB and Good-Turing; excludes no restart.
- **`sequential_comparison.png` / `.pdf`:** detailed three-method comparison.
- **`restart_ablation_bounds.png` / `.pdf`:** restart versus no restart and oracle.
- **`coverage.png` / `.pdf`:** main-comparison simultaneous coverage.
- **`restart_ablation_coverage.png` / `.pdf`:** restart versus no-restart coverage.
- **`summary.json` and `stopping.csv`:** machine-readable results and provenance.

Keep large regenerable arrays in
`artifacts/empirical_bernstein/primary_fixed_fullpop/trajectories.npz`.
Keep interpretation and the suggested paper caption in the
[results write-up](results.md).
Use these stable filenames for reruns; do not create dated plot copies.
