#!/usr/bin/env bash
#SBATCH --job-name=lam-grid
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-49
#SBATCH --output=logs/lam_grid_%A_%a.out
#SBATCH --error=logs/lam_grid_%A_%a.err

# K=100 uniform lambda-grid mixture on ONE discovery setting, chosen by CONFIG.
# Each arm produces two bounds from the same e-process: inverted at log(1/alpha)
# ("uniform_lambda") and at log(1/alpha)+log(t) ("adaptive_lambda").
#
# 1000 trials split into 50 array tasks of 20 trials. Every shard uses the same
# config and seed; trial i always gets SeedSequence(seed).spawn(1000)[i], so the
# merged run equals one unsharded run and reuses the corresponding headline
# pipeline's bootstrap draws for that setting.
#
# Cost: ~129 core-seconds per trial locally (101 lambdas x 2 thresholds x 21
# e-process evaluations at horizon 3000). 20 trials on 8 CPUs is ~5.4 min locally;
# ORCD cores measured ~4.3x slower on job 23685184, so ~23 min per task.
# --time=04:00:00 is ~10x margin.
# Memory: compiled restart statistics are 2 x h^2 float64 (~0.15 GB at h=3000)
# per worker, so 8 workers need ~1.5 GB; 24G is ample.
#
# Submit (RUN_ID must be unique -- shards refuse to overwrite):
#   CONFIG=configs/run_lambda_grid_mixture_bernoulli.json RUN_ID=bernoulli_s888_K100 \
#     sbatch cluster/lambda_grid_mixture_setting_array.sbatch
#
# Merge afterwards with src/combine_lambda_grid_shards.py.

set -euo pipefail

mkdir -p logs artifacts

CONFIG="${CONFIG:?set CONFIG to a configs/run_lambda_grid_mixture_*.json}"
RUN_ID="${RUN_ID:?set RUN_ID}"
N_TASKS="${N_TASKS:-50}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?must run as an array task}"
OUT_DIR="artifacts/lambda_grid_mixture/${RUN_ID}"
PYTHON_CMD="${PYTHON_CMD:-uv run --python 3.11 python}"

echo "START $(date)"
echo "ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-unknown} TASK=${TASK_INDEX}/${N_TASKS}"
echo "CONFIG=${CONFIG} OUT_DIR=${OUT_DIR}"

bash -lc "$PYTHON_CMD src/run_lambda_grid_mixture.py \
  --config ${CONFIG} \
  --task-index ${TASK_INDEX} \
  --n-tasks ${N_TASKS} \
  --output-dir ${OUT_DIR}"

echo "END $(date)"
