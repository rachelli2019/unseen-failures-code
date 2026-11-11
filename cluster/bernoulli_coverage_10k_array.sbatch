#!/usr/bin/env bash
#SBATCH --job-name=bernoulli-10k
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/bernoulli_10k_%A_%a.out
#SBATCH --error=logs/bernoulli_10k_%A_%a.err

# 10,000 trials of the i.i.d. Bernoulli(0.05) stress test at horizon 10,000,
# split across 20 array tasks of 500 trials each.
#
# Why an array: trials are i.i.d., so the trial budget splits cleanly across
# tasks that each run their own seed. 20 tasks x 8 CPUs is 160 cores working
# concurrently instead of 8, turning a ~3.5 h serial job into ~11 min per task.
# Shards are merged afterwards by src/combine_bernoulli_shards.py.
#
# Why these resources: the every-time restart mixture's memory cost depends only
# on the HORIZON, not the trial count -- peak RSS is about 65*h^2 bytes per
# worker, so ~6.5 GB at h=10,000 no matter how many trials a task runs. 8 workers
# therefore need ~52 GB, and --mem=64G covers that with headroom. The 16G in the
# older sbatch templates would OOM at this horizon.
#
# Cost: ~10 core-seconds per trial at h=10,000, so 500 trials on 8 CPUs is
# ~625 s. --time=02:00:00 is ~6x margin for slower cluster cores.

set -euo pipefail

mkdir -p logs artifacts

RUN_ID="${RUN_ID:-h10000_t10000}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-500}"
# Distinct seed per task, so the shards are independent trial streams rather than
# 20 copies of the same 500 trajectories.
SEED=$(( 888000 + TASK_INDEX ))
OUT_DIR="artifacts/bernoulli/${RUN_ID}/task_$(printf '%02d' "${TASK_INDEX}")"
PYTHON_CMD="${PYTHON_CMD:-uv run --python 3.11 python}"

mkdir -p "$OUT_DIR"

echo "START $(date)"
echo "ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-unknown} TASK=${TASK_INDEX} SEED=${SEED}"
echo "OUT_DIR=${OUT_DIR}"

bash -lc "$PYTHON_CMD src/run_bernoulli_simulation.py \
  --p 0.05 \
  --horizon 10000 \
  --n-trials ${TRIALS_PER_TASK} \
  --n-workers 8 \
  --seed ${SEED} \
  --alpha 0.05 \
  --output-json ${OUT_DIR}/summary.json \
  --output-npz  ${OUT_DIR}/trajectories.npz"

echo "END $(date)"
