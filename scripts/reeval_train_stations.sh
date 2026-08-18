#!/bin/bash
# Re-evaluate the headline +mTPI runs on the TRAINING stations at the held-out
# TIME split, i.e. "train stations @ held-out years". The standard eval only
# scores the held-out (spatial-test) stations; this companion measures whether
# TESSERA's edge also holds on the locations the model was fitted at.
#
# What changes vs the standard eval:
#   * --station-split train  -> the SnapshotDownscalingDataset / multi-region
#     snapshot dataset selects stations whose spatial_split == "train"
#     (split="test" is still held-out years). NEW flag, added to evaluate.py;
#     default "test" keeps every other caller's behaviour unchanged.
#   * --output-dir <run>/eval_train_stations -> so the resulting
#     test_summary.json does NOT clobber the held-out one in the run dir.
#
# Fair baseline-vs-TESSERA station matching (the paper convention, same as the
# 898-station held-out table): the no-TESSERA bilinear baseline is re-evaluated
# with --filter-vae-latents-path so its metrics land on exactly the same
# {train ∩ VAE-latent-valid} stations the TESSERA arm sees intrinsically. The
# flag is a no-op for the TESSERA arm (its checkpoint already uses VAE latents),
# so it's only added to the baseline runs.
#
# NO retraining — evaluates the existing best_model.pt checkpoints. evaluate.py
# reads dataset / region / normalisation from the checkpoint config, so only
# --checkpoint (+ the two flags above) is needed. Runs whose
# eval_train_stations/test_summary.json already exists are skipped (idempotent).
#
# Usage (from repo root):
#   bash projects/tessera_downscaling/scripts/reeval_train_stations.sh
# Dry run (print sbatch commands without submitting):
#   DRY_RUN=1 bash projects/tessera_downscaling/scripts/reeval_train_stations.sh
set -euo pipefail

REPO_ROOT="/projects/u6do/pmms2/end-to-end-forecasting"
BASE_DIR="${REPO_ROOT}/projects/tessera_downscaling/.tmp_output"
EVAL_SCRIPT="projects/tessera_downscaling/scripts/evaluate.py"

# VAE lat16 latents used to filter the baseline arm onto the TESSERA station
# set (row-aligned .npy + station CSV, straight from the TESSERA configs).
VAE_NPY="${BASE_DIR}/processed/station_latents_lat16_grad0.5.npy"
VAE_CSV="${BASE_DIR}/processed/tessera_global/station_list_filtered.csv"

SEEDS=(42 123 456)
# Single-region 14y folders (the comparison's REGION_FOLDERS).
REGIONS=(eu us east_asia australia southern_africa)

# "run_name is_baseline" — is_baseline=1 gets the --filter-vae-latents-path
# station match; the TESSERA arms (is_baseline=0) already filter intrinsically.
EXPERIMENTS=(
    "t2m_snap_vae_lat16_concat_with_elev_mtpi_no_static_wd 0"
    "t2m_snap_bilinear_baseline_mtpi_wd 1"
    "wind_truncnormal_snap_vae_lat16_concat_with_elev_mtpi_no_static_wd 0"
    "wind_truncnormal_snap_bilinear_baseline_mtpi_wd 1"
)

BATCH_SIZE=1
NUM_WORKERS=4
TIME="04:00:00"     # eval-only is far cheaper than training
PARTITION=""

mkdir -p "${REPO_ROOT}/logs"

SUBMITTED=0
SKIPPED=0
for reg in "${REGIONS[@]}"; do
    OUTPUT_ROOT="${BASE_DIR}/training_runs_snapshot_14y_${reg}"
    for exp_spec in "${EXPERIMENTS[@]}"; do
        RUN_NAME="${exp_spec% *}"
        IS_BASELINE="${exp_spec##* }"
        for seed in "${SEEDS[@]}"; do
            run_dir="${OUTPUT_ROOT}/${RUN_NAME}_seed${seed}"
            ckpt="${run_dir}/best_model.pt"
            out_dir="${run_dir}/eval_train_stations"
            summary="${out_dir}/test_summary.json"

            if [ ! -f "${ckpt}" ]; then
                echo "SKIP (no checkpoint): ${reg} ${RUN_NAME} seed${seed}"
                SKIPPED=$((SKIPPED + 1)); continue
            fi
            if [ -f "${summary}" ]; then
                echo "SKIP (already done): ${reg} ${RUN_NAME} seed${seed}"
                SKIPPED=$((SKIPPED + 1)); continue
            fi

            job_name="reeval_trainst_${reg}_${RUN_NAME}_s${seed}"
            EVAL_CMD="${REPO_ROOT}/.venv/bin/python ${EVAL_SCRIPT} \
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
                --output=${REPO_ROOT}/logs/${job_name}_%j.out \
                --error=${REPO_ROOT}/logs/${job_name}_%j.err \
                --gpus=1 --time=${TIME} ${PARTITION:+--partition=${PARTITION}} \
                --wrap=\"cd ${REPO_ROOT} && ${EVAL_CMD}\""

            if [ "${DRY_RUN:-0}" = "1" ]; then
                echo "DRY RUN: ${job_name}"
                echo "  ${SBATCH_CMD}"
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
