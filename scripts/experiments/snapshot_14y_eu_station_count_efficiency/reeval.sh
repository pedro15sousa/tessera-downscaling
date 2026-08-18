#!/bin/bash
# Re-evaluate existing trained checkpoints in the station-count
# efficiency experiment with region_specs_test = {"europe":"all"},
# so the per-subset breakdown produces train_stations metrics (the
# model's performance on its own training stations during the
# held-out test year) alongside the existing spatial_test metrics.
#
# Why this exists: when the experiment was first submitted, the eval
# region_specs was {"europe":"test"} — only the 15% held-out spatial
# stations were fed through the model at test time. That gives the
# headline spatial-generalisation number but not the in-network /
# temporal-generalisation number for training stations. submit.sh
# now writes {"europe":"all"}, but the runs already on disk need a
# re-eval pass to produce the new metrics. This script is that pass.
#
# Eval only — no retraining. Existing best_model.pt checkpoints are
# loaded and run through evaluate.py once each; test_summary.json,
# test_predictions.npz, and test_station_errors.npz are overwritten
# in place.
#
# Idempotent: skips runs whose test_summary.json already contains a
# train_stations metric. Pass FORCE=1 to re-run regardless. Skips
# the era5_interp simple-baseline runs (no checkpoint to re-eval —
# they're computed by evaluate_simple_baselines.py instead).
#
# Usage (from repo root):
#   bash projects/tessera_downscaling/scripts/experiments/snapshot_14y_eu_station_count_efficiency/reeval.sh
#
# Dry run:
#   DRY_RUN=1 bash .../reeval.sh
#
# Force re-eval of already-done runs:
#   FORCE=1 bash .../reeval.sh
set -euo pipefail

# ---- Paths (env-overridable, same defaults as submit.sh) ----
REPO_ROOT="${REPO_ROOT:-/projects/u6do/pmms2/end-to-end-forecasting}"
BASE_DIR="${BASE_DIR:-${REPO_ROOT}/projects/tessera_downscaling/.tmp_output}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_DIR}/training_runs_snapshot_14y_eu_station_count_efficiency}"
EVAL_SCRIPT="projects/tessera_downscaling/scripts/evaluate.py"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use the same region_specs_test.json that submit.sh writes (must contain
# {"europe":"all"}). Written here too so reeval.sh works even if
# submit.sh hasn't been run since the all-vs-test flip.
REGION_SPECS_TEST_JSON="${SCRIPT_DIR}/region_specs_test.json"
echo '{"europe":"all"}' > "${REGION_SPECS_TEST_JSON}"

# ---- Slurm settings ----
# Eval is much cheaper than training; smaller GPU-time allocation.
TIME="${TIME:-02:00:00}"
PARTITION="${PARTITION:-}"
BATCH_SIZE=1
NUM_WORKERS=4

FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${REPO_ROOT}/logs"

if [ ! -d "${OUTPUT_ROOT}" ]; then
    echo "ERROR: OUTPUT_ROOT does not exist: ${OUTPUT_ROOT}" >&2
    exit 1
fi

# Cache the current Slurm queue at startup so we can skip jobs that
# are already pending/running. This is critical for re-runs of this
# script: the on-disk test_summary.json check only catches jobs that
# have FINISHED — jobs that are PENDING or RUNNING still show the old
# summary (no train_stations keys), so without the queue check we'd
# happily re-submit them and create duplicates.
#
# Output of `squeue --me -h -o "%j"` is one line per job, just the
# job name. Empty if no jobs in queue. If squeue isn't available
# (e.g. local dry-run), fall through silently and rely on the file
# check only.
QUEUED_JOBS="$(squeue --me -h -o "%j" 2>/dev/null || true)"

JOB_COUNT=0
SKIP_COUNT=0
IN_QUEUE_COUNT=0
NOT_READY_COUNT=0
FAILED_COUNT=0
QOS_HIT=0

for run_dir in "${OUTPUT_ROOT}"/*/; do
    run_name="$(basename "${run_dir}")"

    # Skip simple-baseline run dirs (they hold era5_interp results,
    # not a torch checkpoint — there's nothing for evaluate.py to load).
    if [[ "${run_name}" == *era5_interp_baseline* ]]; then
        continue
    fi

    ckpt="${run_dir}best_model.pt"
    if [ ! -f "${ckpt}" ]; then
        # Training not yet complete — leave it alone. submit.sh will
        # finish it; once test_summary.json exists, reeval.sh on the
        # next run will pick it up.
        NOT_READY_COUNT=$((NOT_READY_COUNT + 1))
        continue
    fi

    job_name="reeval_${run_name}"

    # Idempotent check (a): skip if test_summary.json already has any
    # train_stations metric for any of the run's target variables.
    # The grep covers t2m and wind plus any future variable that
    # follows the {var}_train_stations_* naming convention.
    summary="${run_dir}test_summary.json"
    if [ "${FORCE}" != "1" ] && [ -f "${summary}" ]; then
        if grep -q '"[a-z0-9_]*_train_stations_' "${summary}" 2>/dev/null; then
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi
    fi

    # Idempotent check (b): skip if this job is already in the Slurm
    # queue (pending, running, completing). Without this, re-running
    # reeval.sh after a partial run (e.g. hit QOS limit mid-loop)
    # would re-submit jobs that are still in flight because their
    # test_summary.json hasn't been overwritten yet.
    if [ "${FORCE}" != "1" ] && echo "${QUEUED_JOBS}" | grep -qx "${job_name}"; then
        IN_QUEUE_COUNT=$((IN_QUEUE_COUNT + 1))
        continue
    fi

    EVAL_CMD="${REPO_ROOT}/.venv/bin/python ${EVAL_SCRIPT} \
        --checkpoint ${ckpt} \
        --batch-size ${BATCH_SIZE} \
        --num-workers ${NUM_WORKERS} \
        --region-specs-test-file ${REGION_SPECS_TEST_JSON}"

    SBATCH_CMD="sbatch --job-name=${job_name} \
        --output=${REPO_ROOT}/logs/${job_name}_%j.out \
        --error=${REPO_ROOT}/logs/${job_name}_%j.err \
        --gpus=1 --time=${TIME} \
        ${PARTITION:+--partition=${PARTITION}} \
        --wrap=\"cd ${REPO_ROOT} && ${EVAL_CMD}\""

    if [ "${DRY_RUN}" = "1" ]; then
        echo "DRY RUN: ${job_name}"
        echo "  ${SBATCH_CMD}"
        echo ""
        JOB_COUNT=$((JOB_COUNT + 1))
        continue
    fi

    # Don't bail the whole script on a single sbatch failure (e.g. the
    # QOS submit limit getting hit mid-loop). Capture stdout+stderr,
    # log the failure, and continue — but if the failure is a QOS-limit
    # error, stop trying since every subsequent sbatch in the same
    # invocation will hit the same wall.
    if sbatch_output=$(eval "${SBATCH_CMD}" 2>&1); then
        echo "SUBMITTED: ${job_name} -> ${sbatch_output}"
        JOB_COUNT=$((JOB_COUNT + 1))
    else
        echo "FAILED: ${job_name}"
        echo "  ${sbatch_output}" | head -3
        FAILED_COUNT=$((FAILED_COUNT + 1))
        if echo "${sbatch_output}" | grep -qi "QOS"; then
            QOS_HIT=1
            echo ""
            echo "QOS submit limit hit — stopping early."
            echo "Re-run reeval.sh once the queue has capacity; submissions"
            echo "will be deduplicated against the live queue (no duplicates)."
            break
        fi
    fi
done

echo ""
echo "============================================"
echo "Re-eval submission summary"
echo "============================================"
echo "  Submitted:        ${JOB_COUNT}"
echo "  Already done:     ${SKIP_COUNT}  (test_summary.json has train_stations)"
echo "  Already in queue: ${IN_QUEUE_COUNT}  (skipped to avoid duplicate)"
echo "  Not ready yet:    ${NOT_READY_COUNT}  (no best_model.pt)"
echo "  Failed:           ${FAILED_COUNT}"
echo ""
echo "Re-runs read region_specs from: ${REGION_SPECS_TEST_JSON}"
echo "                                  $(cat "${REGION_SPECS_TEST_JSON}")"
echo "Monitor with: squeue --me -h -o '%T' | sort | uniq -c"
if [ "${QOS_HIT}" = "1" ]; then
    echo ""
    echo "Note: stopped early due to QOS submit limit. Re-run this script"
    echo "after some of your queued jobs finish to submit the rest."
fi