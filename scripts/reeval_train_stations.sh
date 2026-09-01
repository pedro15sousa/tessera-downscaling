#!/bin/bash
# Re-evaluate the headline +mTPI runs on the TRAINING stations at the held-out
# TIME split, i.e. "train stations @ held-out years". The standard eval only
# scores the held-out (spatial-test) stations; this companion measures whether
# TESSERA's edge also holds on the locations the model was fitted at (the
# paper's training-station table, scripts/paper/make_paper_tables.py --tables
# train, and the station counts of Fig. 2).
#
# What changes vs the standard eval:
#   * --station-split train  -> the dataset selects stations whose
#     spatial_split == "train" (split="test" is still held-out years).
#   * --output-dir <run>/eval_train_stations -> so the resulting
#     test_summary.json does NOT clobber the held-out one in the run dir.
#
# Fair baseline-vs-TESSERA station matching (the paper convention, same as the
# held-out table): the no-TESSERA bilinear baseline is re-evaluated with
# --filter-vae-latents-path so its metrics land on exactly the same
# {train ∩ VAE-latent-valid} stations the TESSERA arm sees intrinsically. The
# flag is a no-op for the TESSERA arm (its checkpoint already uses the
# latents), so it's only added to the baseline runs. The latents file MUST be
# the one the TESSERA arm was trained with (its NaN mask is the filter).
#
# NO retraining -- evaluates the existing best_model.pt checkpoints.
# tessera-evaluate reads dataset / region / filters from the checkpoint config,
# so only --checkpoint (+ the flags above) is needed. Runs whose
# eval_train_stations/test_summary.json already exists are skipped unless
# FORCE=1.
#
# Usage (from anywhere; data root from $TESSERA_DATA_ROOT):
#   bash scripts/reeval_train_stations.sh
# Dry run (print sbatch commands without submitting):
#   DRY_RUN=1 bash scripts/reeval_train_stations.sh
# Run in this shell instead of Slurm:
#   LOCAL=1 bash scripts/reeval_train_stations.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${TESSERA_DATA_ROOT:-/data/weather-downscaling}"
export TESSERA_DATA_ROOT="${DATA_ROOT}"
EVAL_CMD_BASE="uv run tessera-evaluate"

# The paper's TESSERA arm: 1B-M (v2) 2017 latents, runs in the
# training_runs/snapshot_14y_<region>${TESSERA_SUFFIX} folders; the no-TESSERA
# baseline lives in training_runs/snapshot_14y_<region>. Same files as
# scripts/experiments/_lib.sh.
TESSERA_SUFFIX="${TESSERA_SUFFIX:-_tessera_1B-M_2017}"
VAE_NPY="${VAE_LATENTS_PATH:-${DATA_ROOT}/processed/vae_tessera_1B-M/station_latents_1B-M_p128_2017_crop64_lat16_grad0.5_auxon.npy}"
VAE_CSV="${VAE_LATENTS_CSV:-${DATA_ROOT}/processed/tessera_global/station_list_filtered.csv}"

SEEDS=(42 123 456)
# Single-region 14y folders (the comparison's REGION_FOLDERS).
REGIONS=(eu us east_asia australia southern_africa)

# "run_name is_baseline" -- is_baseline=1 (no-TESSERA baseline folder) gets the
# --filter-vae-latents-path station match; the TESSERA arms (is_baseline=0,
# TESSERA folder) already filter intrinsically.
EXPERIMENTS=(
    "t2m_snap_vae_crop64_lat16_auxon_concat_mtpi 0"
    "t2m_snap_bilinear_baseline_mtpi_wd 1"
    "wind_truncnormal_snap_vae_crop64_lat16_auxon_concat_mtpi 0"
    "wind_truncnormal_snap_bilinear_baseline_mtpi_wd 1"
)

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TIME="${TIME:-04:00:00}"     # eval-only is far cheaper than training
PARTITION="${PARTITION:-}"
DRY_RUN="${DRY_RUN:-0}"
LOCAL="${LOCAL:-0}"
FORCE="${FORCE:-0}"
LOG_DIR="${REPO_ROOT}/logs"

mkdir -p "${LOG_DIR}"

SUBMITTED=0
SKIPPED=0
for reg in "${REGIONS[@]}"; do
    for exp_spec in "${EXPERIMENTS[@]}"; do
        RUN_NAME="${exp_spec% *}"
        IS_BASELINE="${exp_spec##* }"
        if [ "${IS_BASELINE}" = "1" ]; then
            OUTPUT_ROOT="${DATA_ROOT}/training_runs/snapshot_14y_${reg}"
        else
            OUTPUT_ROOT="${DATA_ROOT}/training_runs/snapshot_14y_${reg}${TESSERA_SUFFIX}"
        fi
        for seed in "${SEEDS[@]}"; do
            run_dir="${OUTPUT_ROOT}/${RUN_NAME}_seed${seed}"
            ckpt="${run_dir}/best_model.pt"
            out_dir="${run_dir}/eval_train_stations"
            summary="${out_dir}/test_summary.json"

            if [ ! -f "${ckpt}" ]; then
                echo "SKIP (no checkpoint): ${reg} ${RUN_NAME} seed${seed}"
                SKIPPED=$((SKIPPED + 1)); continue
            fi
            if [ "${FORCE}" != "1" ] && [ -f "${summary}" ]; then
                echo "SKIP (already done): ${reg} ${RUN_NAME} seed${seed}"
                SKIPPED=$((SKIPPED + 1)); continue
            fi

            job_name="reeval_trainst_${reg}_${RUN_NAME}_s${seed}"
            EVAL_CMD="${EVAL_CMD_BASE} \
                --checkpoint ${ckpt} \
                --station-split train \
                --output-dir ${out_dir} \
                --batch-size ${BATCH_SIZE} --num-workers ${NUM_WORKERS}"
            if [ "${IS_BASELINE}" = "1" ]; then
                EVAL_CMD="${EVAL_CMD} \
                    --filter-vae-latents-path ${VAE_NPY} \
                    --filter-vae-latents-station-csv ${VAE_CSV}"
            fi

            SBATCH_CMD="sbatch --job-name=${job_name} \
                --output=${LOG_DIR}/${job_name}_%j.out \
                --error=${LOG_DIR}/${job_name}_%j.err \
                --gpus=1 --time=${TIME} ${PARTITION:+--partition=${PARTITION}} \
                --wrap=\"cd ${REPO_ROOT} && ${EVAL_CMD}\""

            if [ "${DRY_RUN}" = "1" ]; then
                echo "DRY RUN: ${job_name}"
                echo "  ${SBATCH_CMD}"
            elif [ "${LOCAL}" = "1" ]; then
                echo "RUNNING (local): ${job_name}"
                (cd "${REPO_ROOT}" && eval "${EVAL_CMD}") && echo "DONE: ${job_name}" || echo "FAILED: ${job_name}" >&2
            else
                JOB_ID=$(eval "${SBATCH_CMD}")
                echo "SUBMITTED: ${job_name} -> ${JOB_ID}"
            fi
            SUBMITTED=$((SUBMITTED + 1))
        done
    done
done

echo ""
echo "Train-station re-eval jobs submitted: ${SUBMITTED}   skipped: ${SKIPPED}"
echo "Monitor: squeue --me   |   results -> <run>/eval_train_stations/test_summary.json"
