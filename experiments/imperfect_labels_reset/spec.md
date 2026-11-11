# Perfect labels versus false-negative IPW B-PAC and PPI

**Main experiment:** perfect labels, IPW B-PAC, and PPI with r=0. Use
[the main results](reports_1000/results.md) and [main plot](reports_1000/paper_comparison.png)
for the paper. PPI uses constant audits with probability 0.10; B-PAC uses a
constant 0.10 floor with forced audits outside its certified region. No residual predictor is used in this comparison.

## Population and common settings

Use Q=P: iid sampling with replacement over 4,961 sessions. Unsafe truth means
success score <70 AND at least one judge failure mode: 738 unsafe sessions.
Use the historical 538-category taxonomy and existing Luna binary predictions
and integer uncertainty scores (rescaled from 0–100 to 0–1).

The input snapshot in `data/imperfect_labels/input_snapshot` matches
all five SHA-256 hashes of the previous online-PPI experiment. Missing inputs
and the historical taxonomy were recovered from Git revision `10502f7`.
The current canonical taxonomy has 539 categories and is deliberately untouched.

Alpha=0.05, rho=0.10, seed=888; 1,000 paired trials, 10,000 draws for common bound/coverage plots;
epsilon=0.10,0.125,0.15,0.175,0.20. The 10,000-draw window is used for common bound and coverage plots, not a total
audit budget. Unfinished PPI trials were subsequently continued to stopping;
all current stopping-cost means are complete (see continuation details below).
Use shared session draws and shared uniform audit coins; each arm has its own
history. Repeated sessions are processed again without cached labels.

## Information access and categorization

Assume a correct categorizer: it returns the fixed failure category for an
unsafe session and “safe” for a safe session, including Luna false positives.
The simulation implements this as a lookup, charged as one categorizer call.
For imperfect-label arms, each round executes exactly this sequence:

1. Observe Luna's predicted label and uncertainty. Compute the audit probability
   from the previous calibration state (B-PAC) or rho (PPI).
2. Categorize Luna-positive sessions before the audit. Let mu=1 only if the
   returned category is valid and absent from the pre-draw history. For a Luna
   negative, mu=0 and do not inspect its category.
3. Draw the audit coin and expose truth only if audited. If an audited Luna
   negative is unsafe, categorize it now.
4. Compute the observation, then add a valid category to history only if the
   emitted label is unsafe. The emitted label is audited truth when audited,
   otherwise Luna. An unaudited Luna-negative session exposes neither truth nor
   category and makes no history update.
5. Update B-PAC calibration only after using its pre-round threshold.

The categorizer's “safe” result does not rewrite the binary emitted Luna label;
it prevents a false-positive category from entering discovery/history.
Population frequencies/truth are used separately for diagnostic missing mass.
They never enter routing, mu, calibration, or the stopping decision.

## Three methods

**Perfect:** audit every session; categorize only true-unsafe sessions. The
observation is the binary indicator of a true new category.

**PPI, r=0:** constant audit probability pi=rho=0.10. The plugin mu is defined
above; there is no trained residual or adaptive audit allocation. Let Zstar be
the true new-category indicator relative to this arm's pre-round history.
Use Y=mu+D/pi*(Zstar-mu), without clipping. Zstar is only evaluated on audited
draws. In particular an audited false negative in a new category contributes
1/rho=10, while an unaudited false negative contributes zero. Never redefine
latent Zstar to be the emitted label.

**IPW B-PAC:** zeta=epsilon/2 and delta=alpha/2=0.025. Query with probability rho
below the previous certified uncertainty threshold and probability 1 otherwise
(ties are queried). Initial threshold is -infinity, hence all draws are audited.
On an audited draw use loss 1 only for true-unsafe/Luna-safe pairs:

    L_t(tau) = (1-rho)/pi_t * D_t * false_negative_t * 1{U_t < tau}.
    K_t^kappa(tau) = product_i [1+kappa*(zeta-L_i(tau))].
    kappa_max = 1 / ((1-rho)/rho + zeta).

Average over 100 equally weighted midpoint kappas in (0,kappa_max). Select the
largest passing threshold at level 1/delta, or -infinity if none passes.
Thresholds 0,.01,...,1,+infinity represent every acceptance region for integer
scores, including accepting score 100. The mixture decreases in tau; binary
search requires no alpha split across thresholds. There are no calibration
restarts, no adaptive bets, and no running maximum of calibration wealth.

The discovery observation is Ztilde+zeta, where Ztilde is binary: emitted unsafe
AND valid unseen category. False-positive flags contribute zero because the
categorizer returns “safe.” Discovery uses alpha-delta=0.025.

## Shared discovery mixture and stopping

Use `src/av_core.py` primary log-squared restart weights:
w_s=1/[2.11*(s+1)*log(s+1)^2], with total capital 0.999 and reserve
0.999-sum_{s<=t}w_s. This is the primary implementation's conservative numerical
normalization, rather than the pseudocode's exact unit-mass infinite sum.

Use 100 equally weighted midpoint lambda bets over:
- perfect: (0,1);
- PPI: (0,rho);
- B-PAC discovery: (0,1/(1+zeta)).

Update each restart product with 1+lambda*(u-observation). Stop at the first
M_t(epsilon)>1/a, where a=0.05 for perfect/PPI and 0.025 for B-PAC.
No confidence split across lambda components. Plot bounds rounded upward on a
0.001 grid through 0.30 and a 0.01 grid above; evaluate stopping thresholds and
oracle masses directly, so stopping and coverage are not grid-rounded.

## Reporting and interpretation

Report upper bounds with each arm's actual remaining mass, simultaneous coverage
through the simulation horizon, stopping times, audits through stopping, and
categorizer calls through stopping. Show upper bounds against sessions and
cumulative audits. For matched audit budgets use the first saved 25-draw
checkpoint reaching the budget (at most 24 extra audits); include only budgets
reached by every compared trial. Stopping audit counts are exact.

A shared floor does not imply equal label counts. PPI can consume more sessions
at a fixed audit count. Categorization is also an information/cost source and is
reported separately. These results assume perfect categorization and do not
establish robustness to categorizer errors or an end-to-end label-cost advantage.
The new B-PAC arm calibrates false negatives only; old all-error/adaptive-bet
B-PAC results are not results for this implementation.

## Reproduce

    mkdir -p artifacts/imperfect_labels_reset
    c++ -O3 -std=c++17 -shared -fPIC src/weighted_restart_kernel.cpp -o artifacts/imperfect_labels_reset/weighted_restart.so
    .venv/bin/python -m unittest unit_tests.test_imperfect_labels_reset
    .venv/bin/python src/run_imperfect_labels_reset.py --config configs/run_imperfect_labels_final.json
    .venv/bin/python src/run_imperfect_labels_reset.py --config configs/run_imperfect_labels_ppi_continuation.json
    .venv/bin/python src/complete_ppi_stopping.py
    MPLCONFIGDIR=/tmp/imperfect_labels_mpl .venv/bin/python src/plotting/plot_imperfect_labels_paper.py

The runner refuses to overwrite an existing run. Per-trial metrics are saved
incrementally; configs and input hashes are stored beside the output.

## Main figure presentation

Upper-bound and coverage panels show one B-PAC process with fixed zeta=0.05,
delta=0.025. The stopping-cost sweep uses zeta=epsilon/2, delta=0.025.
Lines show means. Upper-bound shading and label-count error bars show empirical
2.5th–97.5th percentiles across the 1,000 trials, not confidence intervals for
means or simultaneous bands. Ground-truth labels include every forced query
and random audit, including repeated draws. Categorizer calls are separate.
Perfect labels are purple, B-PAC blue, and PPI orange.

## Completion of stopping costs

The original 10,000-draw evaluation left 273 PPI epsilon=.10 trials censored. At the user's request these seeds were replayed and continued with a 20,000-draw safety cap. All stopped by 14,858; no censoring remains in stopping-cost summaries. `src/complete_ppi_stopping.py` verifies exact stored-prefix equality, merges their completed costs with the 727 originally completed trials, and preserves the original 10,000-draw coverage indicators. The original cutoff summaries remain unchanged; main figures use `completed_stopping_summary.json`. Run the continuation config and then this merge script before regenerating the paper plot.

## Release integration

The final runner and tests use the release branch's existing `src/av_core.py`
and native restart kernel, with identical restart-weight functions and byte-identical native kernel to those that
produced the saved results. The wrapper adds optional signed observations while
preserving its default validation. Historical saved-run metadata is retained
unchanged. The new configs use the exact versioned input snapshot above.
No API calls or newly generated labels are needed. Existing legacy PAC scripts
on this branch are separate historical implementations; use the commands above
for the results reported here.
