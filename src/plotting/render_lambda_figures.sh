#!/usr/bin/env bash
# Render every lambda-mixture figure with its final title.
#
# The titles are the reason this script exists: they are editorial, not derivable from
# the data, and every re-render (a rerun, a label change, the full-population fix) would
# otherwise have to reproduce them by hand. Keeping them here makes a re-render one
# command instead of five invocations with five remembered strings.
#
# Run IDs are variables because they change per rerun; override any of them:
#   GRID_PRIMARY=primary_fixed_s888_K100_fullpop ./src/plotting/render_lambda_figures.sh
#
# Figures are written beside the originals with a _lambda_grid suffix; the untitled
# originals are left alone.

set -euo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-uv run python}"
REPORTS=artifacts/reports/e_value_empirical_validation
SUFFIX="${SUFFIX:-_fullpop}"   # headline npz stem suffix; "" for the pre-fix runs
# Full-population figures live under data/plots/full_population/<setting>/; the
# pre-fix figures stay in data/plots/<setting>/ so the two are never mixed.
OUT="${OUT:-data/plots/full_population}"
GRID=artifacts/lambda_grid_mixture
CF=artifacts/closed_form_mixture

GRID_PRIMARY="${GRID_PRIMARY:-primary_fixed_fullpop_s888_K100}"
GRID_ONLINE="${GRID_ONLINE:-online_fullpop_s888_K100}"
GRID_SETVAL="${GRID_SETVAL:-set_valued_fullpop_s888_K100}"
GRID_LLMMERGE="${GRID_LLMMERGE:-llm_merge_fullpop_s888_K100}"
GRID_BERN3K="${GRID_BERN3K:-bernoulli_s888_K100_v2}"
GRID_BERN10K="${GRID_BERN10K:-bernoulli_h10000_s888_K25}"
CF_PRIMARY="${CF_PRIMARY:-primary_fixed_fullpop_s888_cf}"
CF_ONLINE="${CF_ONLINE:-online_s888_cf2}"
CF_SETVAL="${CF_SETVAL:-set_valued_fullpop_s888_cf}"
CF_BERN="${CF_BERN:-bernoulli_s888_cf2}"

headline () {  # <input npz stem> <grid run id> <output dir> <title> [extra flags...]
  local stem="$1" run="$2" out="$3" title="$4"; shift 4
  $PY src/plotting/plot_av_pipeline.py \
    --input-npz "$REPORTS/$stem.npz" \
    --ours-npz "$GRID/$run/combined/trajectories.npz" \
    --ours-key bound__uniform_K100 \
    --output-suffix _lambda_grid --output-dir "$out" --ci-level 0.95 \
    --title "$title" "$@"
}

comparison () {  # <grid run id> <closed-form run id> <prefix> <setting label> [extra flags...]
  local grid="$1" cf="$2" prefix="$3" label="$4"; shift 4
  $PY src/plotting/plot_lambda_comparison.py \
    --grid-npz "$GRID/$grid/combined/trajectories.npz" \
    --closed-form-npz "$CF/$cf/combined/trajectories.npz" \
    --setting-label "$label" \
    --output-dir data/plots/lambda_comparison --prefix "$prefix" "$@"
}

# Only the primary-fixed figure is titled; the others are used with captions.
echo "== headline bound figures =="
headline "primary_oneshot_fixed$SUFFIX"    "$GRID_PRIMARY"  "$OUT/primary_oneshot_fixed"     "Upper Bounds on SWE-chat"
headline "primary_oneshot_online$SUFFIX"   "$GRID_ONLINE"   "$OUT/primary_oneshot_online"    "" --hide-gt
headline "set_valued_oneshot_fixed$SUFFIX" "$GRID_SETVAL"   "$OUT/set_valued_oneshot_fixed"  ""
headline "primary_llm_merge_fixed$SUFFIX"  "$GRID_LLMMERGE" "$OUT/primary_llm_merge_fixed"   ""

echo "== Bernoulli coverage (K=25, horizon 10,000; +/-2 SE band clears 0.95) =="
$PY src/plotting/plot_bernoulli_coverage.py \
  --input-npz "$GRID/$GRID_BERN10K/combined/trajectories.npz" \
  --bound-key bound__uniform_K25 --se-multiplier 2 \
  --output-dir data/plots/bernoulli --title "Coverage on Synthetic Data"

echo "== four-arm lambda comparisons =="
# The fixed-lambda arm comes from the grid run's own single_lambda_0p25, not the
# lambda_restart_ablation npz -- that one predates the full-population fix and its draws
# no longer match (the plotter refuses such a pairing rather than plotting it).
comparison "$GRID_PRIMARY" "$CF_PRIMARY" primary_fixed "Primary fixed taxonomy"
# online + llm_merge have no full-population closed-form run yet, so no comparison figure.
comparison "$GRID_SETVAL"  "$CF_SETVAL"  set_valued  "Set-valued failure modes"
comparison "$GRID_BERN3K"  "$CF_BERN"    bernoulli   "I.i.d. Bernoulli, p = 0.05"

echo "== done =="
