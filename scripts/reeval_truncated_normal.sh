#!/bin/bash
# Re-evaluate every trained checkpoint that uses a truncated-normal head,
# so its test_summary.json picks up the renamed point-estimate metrics:
#
#     <var>_mae   -> <var>_mae_at_median   (MAE now on the predictive MEDIAN)
#     <var>_rmse  -> <var>_rmse_at_mean     (RMSE on the predictive MEAN)
#     <var>_bias  -> <var>_bias_at_mean
#     <var>_correlation -> <var>_correlation_at_mean
#
# WHY: the truncated normal is right-skewed near calm, so MAE should be
# reported on the median (minimises expected |error|) and RMSE on the mean,
# matching the Weibull head. The old evaluator reported both on the mean.
# This is purely a metric-reporting change — NO retraining, the model
# weights are untouched. evaluate.py reloads best_model.pt and rewrites
# test_summary.json / test_results.json / test_predictions.npz /
# test_station_errors.npz in place.
#
# (The notebook helpers also alias the old keys to the new names when they
# load a stale file, so analysis keeps working even before a run is
# re-evaluated. This script makes the on-disk numbers *correct* — the alias
# is only an approximation for the MAE column, since the true median MAE
# can't be recovered from the mean-based value.)
#
# DETECTION: a run is a target iff its config.json's likelihood_per_variable
# assigns "truncated_normal" to any variable (grep on the literal string).
#
# REGION SPECS: if the run's experiment folder carries a
# region_specs_test.json (the EU efficiency / rollout folders do), it is
# passed through with --region-specs-test-file so the same station set is
# scored as in the original eval. Folders without one (the standard
# regional folders: australia, east_asia, southern_africa, eu, us,
# aurora_zeroshot, ...) called evaluate.py with --checkpoint only, which
# this script reproduces exactly.
#
# IDEMPOTENT: skips runs whose test_summary.json already contains a
# *_mae_at_median key (i.e. already re-evaluated under the new code). Pass
# FORCE=1 to re-run regardless. Also dedups against the live Slurm queue.
#
# LIMITATION: this reproduces the *standard* per-folder evaluation that
# writes into the run directory. If a checkpoint was ALSO evaluated into a
# separate output dir with bespoke flags (e.g. an Aurora lead-time eval
# using --dataset-dir/--output-dir), those flags aren't recorded in
# config.json and are NOT reconstructed here — re-run those by hand. The
# script prints a warning listing any nested eval subdirs it detects with
# stale truncated-normal metrics.
#
# Usage (from repo root):
#   bash projects/tessera_downscaling/scripts/reeval_truncated_normal.sh
#
# Dry run (print the commands, submit nothing):
#   DRY_RUN=1 bash .../reeval_truncated_normal.sh
#
# Run locally instead of via Slurm (no sbatch; runs each eval in-process):
#   LOCAL=1 bash .../reeval_truncated_normal.sh
#
# Force re-eval of already-done runs:
#   FORCE=1 bash .../reeval_truncated_normal.sh
#
# Restrict to specific output roots:
#   ROOTS="training_runs_snapshot_14y_australia training_runs_snapshot_14y_eu" \
#       bash .../reeval_truncated_normal.sh
set -euo pipefail

# ---- Paths (env-overridable, same defaults as the per-folder submit.sh) ----
REPO_ROOT="${REPO_ROOT:-/projects/u6do/pmms2/end-to-end-forecasting}"
BASE_DIR="${BASE_DIR:-${REPO_ROOT}/projects/tessera_downscaling/.tmp_output}"
EXPERIMENTS_DIR="${REPO_ROOT}/projects/tessera_downscaling/scripts/experiments"
EVAL_SCRIPT="projects/tessera_downscaling/scripts/evaluate.py"

# ---- Slurm settings (eval is cheap vs training) ----
TIME="${TIME:-02:00:00}"
PARTITION="${PARTITION:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"

FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
LOCAL="${LOCAL:-0}"

mkdir -p "${REPO_ROOT}/logs"

# Resolve the set of output roots to scan. Default: every training_runs_*
# directory under BASE_DIR. Override with ROOTS (space-separated basenames
# or absolute paths).
if [ -n "${ROOTS:-}" ]; then
    SCAN_ROOTS=()
    for r in ${ROOTS}; do
        if [[ "${r}" = /* ]]; then
            SCAN_ROOTS+=("${r}")
        else
            SCAN_ROOTS+=("${BASE_DIR}/${r}")
        fi
    done
else
    SCAN_ROOTS=()
    for d in "${BASE_DIR}"/training_runs_*/; do
        [ -d "${d}" ] && SCAN_ROOTS+=("${d%/}")
    done
fi

if [ "${#SCAN_ROOTS[@]}" -eq 0 ]; then
    echo "ERROR: no training_runs_* roots found under ${BASE_DIR}" >&2
    exit 1
fi

# Cache the live Slurm queue so we don't resubmit jobs already pending /
# running (their test_summary.json still shows the old keys until they
# finish). Empty / silent if squeue is unavailable (e.g. LOCAL runs).
QUEUED_JOBS="$(squeue --me -h -o "%j" 2>/dev/null || true)"

JOB_COUNT=0
SKIP_DONE=0
SKIP_NOT_TN=0
IN_QUEUE_COUNT=0
NOT_READY_COUNT=0
FAILED_COUNT=0
QOS_HIT=0
declare -a NESTED_WARNINGS=()

for OUTPUT_ROOT in "${SCAN_ROOTS[@]}"; do
    [ -d "${OUTPUT_ROOT}" ] || continue
    root_name="$(basename "${OUTPUT_ROOT}")"
    # training_runs_<folder> -> <folder>; used to find the region-specs file.
    folder="${root_name#training_runs_}"
    region_specs_file="${EXPERIMENTS_DIR}/${folder}/region_specs_test.json"

    for run_dir in "${OUTPUT_ROOT}"/*/; do
        [ -d "${run_dir}" ] || continue
        run_name="$(basename "${run_dir}")"

        # Simple-baseline run dirs hold interpolation results, not a torch
        # checkpoint — nothing for evaluate.py to load.
        if [[ "${run_name}" == *era5_interp_baseline* ]] \
           || [[ "${run_name}" == *bilinear_baseline* && ! -f "${run_dir}best_model.pt" ]]; then
            continue
        fi

        config="${run_dir}config.json"
        # Only target truncated-normal runs.
        if [ ! -f "${config}" ] || ! grep -q '"truncated_normal"' "${config}" 2>/dev/null; then
            SKIP_NOT_TN=$((SKIP_NOT_TN + 1))
            continue
        fi

        # Detect (but don't auto-fix) bespoke nested eval subdirs with stale
        # truncated-normal metrics — these were produced with flags we can't
        # reconstruct from config.json.
        while IFS= read -r nested; do
            [ -z "${nested}" ] && continue
            if grep -q '"truncated_normal"' "${nested}" 2>/dev/null \
               && ! grep -q '_mae_at_median' "${nested}" 2>/dev/null; then
                NESTED_WARNINGS+=("${nested}")
            fi
        done < <(find "${run_dir}" -mindepth 2 -name test_summary.json 2>/dev/null)

        ckpt="${run_dir}best_model.pt"
        if [ ! -f "${ckpt}" ]; then
            NOT_READY_COUNT=$((NOT_READY_COUNT + 1))
            continue
        fi

        job_name="reeval_tn_${run_name}"

        # Idempotent: skip if the top-level summary already has a
        # *_mae_at_median key (i.e. already re-evaluated under new code).
        summary="${run_dir}test_summary.json"
        if [ "${FORCE}" != "1" ] && [ -f "${summary}" ] \
           && grep -q '_mae_at_median' "${summary}" 2>/dev/null; then
            SKIP_DONE=$((SKIP_DONE + 1))
            continue
        fi

        # Dedup against the live queue.
        if [ "${FORCE}" != "1" ] && echo "${QUEUED_JOBS}" | grep -qx "${job_name}"; then
            IN_QUEUE_COUNT=$((IN_QUEUE_COUNT + 1))
            continue
        fi

        # Build the eval command. Auto-attach region-specs if the folder
        # has one (efficiency / rollout folders); otherwise --checkpoint
        # only, matching the standard regional submit.sh.
        EVAL_CMD="${REPO_ROOT}/.venv/bin/python ${EVAL_SCRIPT} \
            --checkpoint ${ckpt} \
            --batch-size ${BATCH_SIZE} \
            --num-workers ${NUM_WORKERS}"
        region_note=""
        if [ -f "${region_specs_file}" ]; then
            EVAL_CMD="${EVAL_CMD} --region-specs-test-file ${region_specs_file}"
            region_note=" [region-specs: $(cat "${region_specs_file}")]"
        fi

        if [ "${DRY_RUN}" = "1" ]; then
            echo "DRY RUN: ${job_name}${region_note}"
            echo "  ${EVAL_CMD}"
            JOB_COUNT=$((JOB_COUNT + 1))
            continue
        fi

        if [ "${LOCAL}" = "1" ]; then
            # Run in-process, sequentially. Don't abort the whole sweep on a
            # single failure.
            echo "RUN (local): ${job_name}${region_note}"
            if ( cd "${REPO_ROOT}" && eval "${EVAL_CMD}" ); then
                JOB_COUNT=$((JOB_COUNT + 1))
            else
                echo "FAILED: ${job_name}"
                FAILED_COUNT=$((FAILED_COUNT + 1))
            fi
            continue
        fi

        SBATCH_CMD="sbatch --job-name=${job_name} \
            --output=${REPO_ROOT}/logs/${job_name}_%j.out \
            --error=${REPO_ROOT}/logs/${job_name}_%j.err \
            --gpus=1 --time=${TIME} \
            ${PARTITION:+--partition=${PARTITION}} \
            --wrap=\"cd ${REPO_ROOT} && ${EVAL_CMD}\""

        if sbatch_output=$(eval "${SBATCH_CMD}" 2>&1); then
            echo "SUBMITTED: ${job_name}${region_note} -> ${sbatch_output}"
            JOB_COUNT=$((JOB_COUNT + 1))
        else
            echo "FAILED: ${job_name}"
            echo "  ${sbatch_output}" | head -3
            FAILED_COUNT=$((FAILED_COUNT + 1))
            if echo "${sbatch_output}" | grep -qi "QOS"; then
                QOS_HIT=1
                echo ""
                echo "QOS submit limit hit — stopping early. Re-run after the"
                echo "queue drains; submissions dedup against the live queue."
                break 2
            fi
        fi
    done
done

echo ""
echo "============================================"
echo "Truncated-normal re-eval summary"
echo "============================================"
echo "  Submitted/ran:     ${JOB_COUNT}"
echo "  Already done:      ${SKIP_DONE}  (test_summary.json has *_mae_at_median)"
echo "  Not truncated-norm:${SKIP_NOT_TN}  (config has no truncated_normal head)"
echo "  Already in queue:  ${IN_QUEUE_COUNT}"
echo "  Not ready yet:     ${NOT_READY_COUNT}  (no best_model.pt)"
echo "  Failed:            ${FAILED_COUNT}"

if [ "${#NESTED_WARNINGS[@]}" -gt 0 ]; then
    echo ""
    echo "WARNING: ${#NESTED_WARNINGS[@]} nested eval dir(s) have stale"
    echo "truncated-normal metrics but were produced with bespoke flags"
    echo "(e.g. --dataset-dir/--output-dir). Re-run these manually:"
    for n in "${NESTED_WARNINGS[@]}"; do
        echo "  - ${n}"
    done
fi

if [ "${QOS_HIT}" = "1" ]; then
    echo ""
    echo "Note: stopped early due to QOS submit limit; re-run to submit the rest."
fi
echo ""
echo "Monitor with: squeue --me -h -o '%T' | sort | uniq -c"
